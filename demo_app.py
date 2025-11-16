import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
import pickle
import time
import random
import io
import contextlib
import base64
import traceback
import json
import requests
from pathlib import Path
from datetime import date

# --- AI & API Clients ---
from openai import OpenAI, APIError
from elevenlabs import save
from elevenlabs.client import ElevenLabs

# ============================================================================
# SECTION 1: CONFIGURATION & CONSTANTS
# ============================================================================
class Config:
    DATA_DIR = Path("./data")
    SALES_FILE = DATA_DIR / "valio_aimo_sales_and_deliveries_junction_2025.csv"
    PRODUCT_FILE = DATA_DIR / "valio_aimo_product_data_junction_2025.json"
    REPLACEMENT_FILE = DATA_DIR / "valio_aimo_replacement_orders_junction_2025.csv"
    MODEL_PATH = Path("shortage_predictor_model.pkl")
    AUDIO_DIR = Path("generated_audio")
    CRITICAL_RISK_THRESHOLD = 0.8
    HIGH_RISK_THRESHOLD = 0.5
    MEDIUM_RISK_THRESHOLD = 0.3

MOCK_PRODUCT_IMAGES = {
    "405983": {"name": "Frutica", "url": "https://placehold.co/150x150/E91E63/white?text=Yogurt"},
    "405432": {"name": "Speed", "url": "https://placehold.co/150x150/2196F3/white?text=Milk"},
}
# ============================================================================
# SECTION 2: CORE LOGIC & AI HELPERS
# ============================================================================
class ShortagePredictor:
    def __init__(self, model_path=None):
        if model_path is None: model_path = Config.MODEL_PATH
        with open(model_path, 'rb') as f: artifacts = pickle.load(f)
        self.model = artifacts['model']
        self.feature_cols = artifacts['feature_cols']
        fi_df = artifacts.get('feature_importance', pd.DataFrame({'feature': [], 'importance': []}))
        self.feature_importance_map = dict(zip(fi_df['feature'], fi_df['importance']))
        self.threshold = artifacts.get('threshold', 0.5)

    def load_historical_data(self, df_features):
        self.product_stats = df_features.group_by('product_code').agg([
            pl.col('is_shortage').mean().alias('product_shortage_rate'), pl.col('order_qty').mean().alias('product_avg_order_qty'),
            pl.col('order_qty').std().alias('product_order_qty_std'), pl.col('order_number').n_unique().alias('product_order_count')
        ])
        self.customer_stats = df_features.group_by('customer_number').agg([
            pl.col('is_shortage').mean().alias('customer_shortage_rate'), pl.col('order_qty').sum().alias('customer_total_order_qty'),
            pl.col('order_number').n_unique().alias('customer_order_count'), pl.col('lead_time_days').mean().alias('customer_avg_lead_time')
        ])
        self.customer_product_stats = df_features.group_by(['customer_number', 'product_code']).agg([
            pl.col('is_shortage').mean().alias('customer_product_shortage_rate'), pl.col('order_number').n_unique().alias('customer_product_order_count')
        ])
        self.plant_stats = df_features.group_by('plant').agg([pl.col('is_shortage').mean().alias('plant_shortage_rate')])
        self.storage_stats = df_features.group_by('storage_location').agg([pl.col('is_shortage').mean().alias('storage_shortage_rate')])
        self.category_stats = df_features.group_by('category').agg([pl.col('is_shortage').mean().alias('category_shortage_rate')])
        self.temp_stats = df_features.group_by('temperatureCondition').agg([pl.col('is_shortage').mean().alias('temp_condition_shortage_rate')])
        self.vendor_stats = df_features.group_by('vendorName').agg([pl.col('is_shortage').mean().alias('vendor_shortage_rate'), pl.col('order_number').n_unique().alias('vendor_order_count')])

    def engineer_features(self, df_new_orders):
        df = df_new_orders.clone()
        if 'product_code' in df.columns: df = df.with_columns([pl.col('product_code').cast(pl.Utf8)])
        df = df.with_columns([
            pl.col('order_created_date').dt.weekday().alias('order_day_of_week'), pl.col('order_created_date').dt.day().alias('order_day_of_month'),
            pl.col('order_created_date').dt.month().alias('order_month'), pl.col('order_created_date').dt.quarter().alias('order_quarter'),
            (pl.col('order_created_date').dt.weekday() >= 5).cast(pl.Int8).alias('is_weekend'),
        ])
        if 'requested_delivery_date' not in df.columns: df = df.with_columns([(pl.col('order_created_date') + pl.duration(days=1)).alias('requested_delivery_date')])
        df = df.with_columns([((pl.col('requested_delivery_date') - pl.col('order_created_date')).dt.total_days()).alias('lead_time_days')])
        if 'order_created_time' in df.columns: df = df.with_columns([(pl.col('order_created_time') // 10000).alias('order_hour')])
        else: df = df.with_columns([pl.lit(14).alias('order_hour')])
        if 'picking_picked_qty' not in df.columns: df = df.with_columns([pl.lit(0).alias('picking_picked_qty')])
        df = df.join(self.product_stats, on='product_code', how='left')
        df = df.join(self.customer_stats, on='customer_number', how='left')
        df = df.join(self.customer_product_stats, on=['customer_number', 'product_code'], how='left')
        df = df.join(self.plant_stats, on='plant', how='left')
        df = df.join(self.storage_stats, on='storage_location', how='left')
        GLOBAL_SHORTAGE_RATE = 0.04
        df = df.with_columns([
            pl.col('product_shortage_rate').fill_null(GLOBAL_SHORTAGE_RATE), pl.col('product_avg_order_qty').fill_null(pl.col('order_qty')),
            pl.col('product_order_qty_std').fill_null(0), pl.col('product_order_count').fill_null(0),
            pl.col('customer_shortage_rate').fill_null(GLOBAL_SHORTAGE_RATE), pl.col('customer_total_order_qty').fill_null(0),
            pl.col('customer_order_count').fill_null(0), pl.col('customer_avg_lead_time').fill_null(pl.col('lead_time_days')),
            pl.col('customer_product_shortage_rate').fill_null(GLOBAL_SHORTAGE_RATE), pl.col('customer_product_order_count').fill_null(0),
            pl.col('plant_shortage_rate').fill_null(GLOBAL_SHORTAGE_RATE), pl.col('storage_shortage_rate').fill_null(GLOBAL_SHORTAGE_RATE),
        ])
        for col in ['vendor_shortage_rate', 'vendor_order_count', 'replacement_frequency', 'category_shortage_rate', 'temp_condition_shortage_rate']:
            if col not in df.columns:
                default_val = GLOBAL_SHORTAGE_RATE if 'rate' in col else 0
                df = df.with_columns([pl.lit(default_val).alias(col)])
        return df

    def predict(self, df_new_orders):
        df_features = self.engineer_features(df_new_orders)
        X = df_features.select(self.feature_cols).to_numpy()
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        df_result = df_features.with_columns([pl.lit(predictions).alias('predicted_shortage'), pl.lit(probabilities).alias('shortage_probability')])
        return df_result

    def explain_prediction_as_string(self, df_row):
        features = df_row.select(self.feature_cols).to_numpy()[0]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print("📊 Top Risk Factors:")
            feature_values = {}
            for i, col in enumerate(self.feature_cols):
                importance = self.feature_importance_map.get(col, 0)
                feature_values[col] = {'value': features[i], 'importance': importance, 'contribution': features[i] * importance}
            sorted_features = sorted(feature_values.items(), key=lambda x: abs(x[1]['contribution']), reverse=True)
            for i, (feat, info) in enumerate(sorted_features[:10], 1):
                print(f"  {i:2d}. {feat:.<40} {info['value']:.4f}")
        return output.getvalue()

class SubstitutionEngine:
    def __init__(self, df_products, df_replacements=None):
        self.df_products = df_products
        self.df_replacements = df_replacements
        self.substitution_matrix = self._build_substitution_matrix()
    def _build_substitution_matrix(self):
        substitutions = {}
        if self.df_products is not None and 'category' in self.df_products.columns:
            category_groups = self.df_products.group_by('category').agg([pl.col('product_code').alias('products')])
            for row in category_groups.iter_rows(named=True):
                products = row['products']
                for prod in products:
                    if prod not in substitutions: substitutions[prod] = []
                    substitutions[prod].extend([p for p in products if p != prod])
        if self.df_replacements is not None and self.df_replacements.height > 0:
            replacement_products = self.df_replacements['product_code'].cast(pl.Utf8).unique().to_list()
            for prod in replacement_products:
                if prod not in substitutions: substitutions[prod] = []
        return substitutions
    def get_substitutes(self, product_code, top_n=3):
        product_code = str(product_code)
        if product_code not in self.substitution_matrix: return []
        substitutes = self.substitution_matrix[product_code]
        return list(dict.fromkeys(substitutes))[:top_n]

def run_smart_dispatcher(predictor, sub_engine, input_orders_df):
    if 'order_created_time' not in input_orders_df.columns: input_orders_df = input_orders_df.with_columns([pl.lit(140000).alias('order_created_time')])
    if 'requested_delivery_date' not in input_orders_df.columns: input_orders_df = input_orders_df.with_columns([(pl.col('order_created_date') + pl.duration(days=1)).alias('requested_delivery_date')])
    predictions = predictor.predict(input_orders_df)
    predictions = predictions.with_columns([
        pl.when(pl.col('shortage_probability') >= Config.CRITICAL_RISK_THRESHOLD).then(pl.lit('🔴 CRITICAL')).when(pl.col('shortage_probability') >= Config.HIGH_RISK_THRESHOLD).then(pl.lit('🟡 HIGH')).when(pl.col('shortage_probability') >= Config.MEDIUM_RISK_THRESHOLD).then(pl.lit('🟠 MEDIUM')).otherwise(pl.lit('🟢 LOW')).alias('risk_level')
    ])
    high_priority = predictions.filter(pl.col('risk_level').is_in(['🔴 CRITICAL', '🟡 HIGH'])).sort('shortage_probability', descending=True)
    if high_priority.height > 0:
        actions = []
        for row in high_priority.iter_rows(named=True):
            substitutes = sub_engine.get_substitutes(str(row['product_code']))
            action = '📞 Offer substitutes' if substitutes else '📞 Notify of potential delay'
            actions.append({
                'order_number': row['order_number'], 'risk_level': row['risk_level'], 'product_code': row['product_code'],
                'customer_number': row['customer_number'], 'order_qty': row['order_qty'], 'shortage_probability': row['shortage_probability'],
                'recommended_action': action, 'substitutes': ", ".join(substitutes) if substitutes else "None Found"
            })
        return predictions, pl.DataFrame(actions)
    else:
        return predictions, None

@st.cache_data
def generate_call_script(order_details: dict, mock_mode=False) -> str:
    if mock_mode:
        return f"This is a mock call script for customer {order_details['customer_number']} about product {order_details['product_code']}. We suggest substitutes: {order_details['substitutes']}."
    try:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=st.secrets["OPENROUTER_API_KEY"])
        substitutes_text = f"We have identified these potential substitutes for you: {order_details['substitutes']}." if order_details['substitutes'] != "None Found" else "Unfortunately, we don't have an immediate substitute recommendation."
        prompt = f"You are a friendly and proactive logistics assistant from Valio Aimo. Your goal is to inform a customer about a potential shortage and offer a solution. Generate a short, professional, and friendly script for a phone call based on the following details. Keep it under 60 words. - Customer Number: {order_details['customer_number']} - Product at Risk (Code): {order_details['product_code']} - Quantity Ordered: {order_details['order_qty']} - Recommended Action: {order_details['recommended_action']} - Available Substitutes: {substitutes_text}. Start the call with 'Hi, this is a proactive notification from Valio Aimo regarding your recent order.'"
        if st.session_state.get("debug_mode", False):
            with st.expander("🤖 View Text Generation Prompt"):
                st.code(prompt, language="text")
        response = client.chat.completions.create(model="openai/gpt-oss-120b", messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content.strip()
    except APIError as e:
        st.error(f"OpenRouter API Error: {e.status_code}. Response: {e.response.text}")
        return None
    except Exception:
        st.error("An unexpected error occurred with OpenRouter.")
        st.code(traceback.format_exc())
        return None

@st.cache_data
def generate_audio_from_script(script_text: str, order_number: int, mock_mode=False) -> str:
    file_path = str(Config.AUDIO_DIR / f"{order_number}.mp3")
    if mock_mode:
        st.info("API Mocked: Returning path to dummy audio file.")
        return file_path
    try:
        client = ElevenLabs(api_key=st.secrets["ELEVENLABS_API_KEY"])
        audio_iterator = client.text_to_speech.convert(voice_id="21m00Tcm4TlvDq8ikWAM", text=script_text, model_id="eleven_multilingual_v2")
        Config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        save(audio_iterator, file_path)
        return file_path
    except APIError as e:
        st.error(f"ElevenLabs API Error: {e.status_code}. Response: {e.response.text}")
        return None
    except Exception:
        st.error("An unexpected error occurred with ElevenLabs.")
        st.code(traceback.format_exc())
        return None

@st.cache_data
def identify_product_in_image(image_bytes, mock_mode=False):
    debug_log = []
    if mock_mode:
        return random.choice(list(MOCK_PRODUCT_IMAGES.keys())), "Mock Mode: Returned random product."
    candidate_list = "\n".join([f"- ID: {pid}, Name: {pinfo['name']}" for pid, pinfo in MOCK_PRODUCT_IMAGES.items()])
    prompt = f"Analyze the user's uploaded image. From the following list of candidate products, determine which product is shown. Respond ONLY with the matching product ID (e.g., '405983'). If you cannot confidently identify a match from the list, respond ONLY with the word 'NONE'.\n\nCandidate Products:\n{candidate_list}"
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    try:
        debug_log.append("--- Attempt 1: Using official OpenAI client ---")
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=st.secrets["OPENROUTER_API_KEY"])
        if st.session_state.get("debug_mode", False):
            with st.expander("🤖 View Vision Prompt (Attempt 1)"):
                st.code(prompt, language="text")
        response = client.chat.completions.create(model="nvidia/nemotron-nano-12b-v2-vl", messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}], max_tokens=128, temperature=0.0)
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            result_text = response.choices[0].message.content.strip()
            debug_log.append(f"Success with OpenAI client. AI response: '{result_text}'")
            if result_text in MOCK_PRODUCT_IMAGES:
                return result_text, "\n".join(debug_log)
            debug_log.append("AI response was not a known product ID.")
        else:
            debug_log.append("AI returned an empty or invalid response via OpenAI client.")
        return None, "\n".join(debug_log)
    except (TypeError, Exception) as e:
        debug_log.append(f"Official OpenAI client failed. Error: {type(e).__name__}: {e}")
        debug_log.append("\n--- Attempt 2: Falling back to `requests` library ---")
        try:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {"Authorization": f"Bearer {st.secrets['OPENROUTER_API_KEY']}", "Content-Type": "application/json"}
            payload = {"model": "nvidia/nemotron-nano-12b-v2-vl", "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}], "max_tokens": 128, "temperature": 0.0}
            debug_log.append(f"Requesting URL: {url}")
            debug_log.append(f"Request Payload:\n{json.dumps(payload, indent=2)}")
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            debug_log.append(f"Raw Response Status Code: {response.status_code}")
            debug_log.append(f"Raw Response Body:\n{response.text}")
            response.raise_for_status()
            data = response.json()
            if data.get("choices") and data["choices"][0].get("message") and data["choices"][0]["message"].get("content"):
                result_text = data["choices"][0]["message"]["content"].strip()
                debug_log.append(f"Success with `requests` fallback. AI response: '{result_text}'")
                if result_text in MOCK_PRODUCT_IMAGES:
                    return result_text, "\n".join(debug_log)
            return None, "\n".join(debug_log)
        except Exception as fallback_e:
            debug_log.append(f"Fallback with `requests` also failed. Error: {type(fallback_e).__name__}: {fallback_e}")
            st.error("Vision AI failed with both methods. Check the debug log for details.")
            st.code(traceback.format_exc())
            return None, "\n".join(debug_log)
# ============================================================================
# SECTION 3: STREAMLIT APP LOGIC
# ============================================================================
@st.cache_resource
def load_dependencies():
    if "OPENROUTER_API_KEY" not in st.secrets or "ELEVENLABS_API_KEY" not in st.secrets:
        st.error("🚨 API keys for OpenRouter and/or ElevenLabs are not configured. Please add them to your .streamlit/secrets.toml file.")
        st.stop()
    if not Config.MODEL_PATH.exists():
        st.error(f"FATAL: Model file not found at '{Config.MODEL_PATH}'. Please run the training script first.")
        st.stop()
    predictor = ShortagePredictor(Config.MODEL_PATH)
    df_sales = pl.read_csv(Config.SALES_FILE, try_parse_dates=True).with_columns(pl.col("product_code").cast(pl.Utf8))
    df_clean = df_sales.with_columns([pl.col('delivered_qty').fill_null(0)])
    df_clean = df_clean.with_columns([((pl.col('order_qty') > pl.col('delivered_qty')) | ((pl.col('delivered_qty') == 0) & (pl.col('order_qty') > 0))).cast(pl.Int8).alias('is_shortage')])
    df_clean = df_clean.with_columns([pl.lit('UNKNOWN').alias('vendorName'), pl.lit('UNKNOWN').alias('category'), pl.lit('UNKNOWN').alias('temperatureCondition')])
    if 'requested_delivery_date' not in df_clean.columns: df_clean = df_clean.with_columns([(pl.col('order_created_date') + pl.duration(days=1)).alias('requested_delivery_date')])
    df_clean = df_clean.with_columns([((pl.col('requested_delivery_date') - pl.col('order_created_date')).dt.total_days()).alias('lead_time_days')])
    predictor.load_historical_data(df_clean)
    df_products = pl.DataFrame({'product_code': ['STUB'], 'category': ['UNKNOWN']})
    df_replacements = pl.read_csv(Config.REPLACEMENT_FILE).with_columns(pl.col("product_code").cast(pl.Utf8))
    sub_engine = SubstitutionEngine(df_products, df_replacements)
    sample_orders = df_clean.sort('order_created_date', descending=True).head(1000).select(['order_number', 'order_created_date', 'customer_number', 'product_code', 'order_qty', 'plant', 'storage_location', 'requested_delivery_date', 'order_created_time'])
    return predictor, sub_engine, sample_orders, df_clean

# --- Main App UI ---
st.set_page_config(layout="wide", page_title="Valio Aimo Smart Dispatcher")

# --- Initialize Session State ---
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False
if "dispatcher_run" not in st.session_state:
    st.session_state.dispatcher_run = False
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "actions" not in st.session_state:
    st.session_state.actions = None
if "notification_details" not in st.session_state:
    st.session_state.notification_details = {}

# --- Developer Sidebar ---
st.sidebar.header("⚙️ Developer Options")
st.session_state.debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=st.session_state.debug_mode)
mock_apis = st.sidebar.checkbox("Mock API Calls (No Cost)", value=True)
if st.session_state.debug_mode:
    st.sidebar.subheader("🕵️ Session State Inspector")
    st.sidebar.json(st.session_state.to_dict(), expanded=False)

# Load Dependencies
predictor, sub_engine, sample_orders, historical_data = load_dependencies()

# --- App Header and Model Performance ---
st.title("🚀 Valio Aimo - Smart Dispatcher AI")
st.markdown("This application uses a pre-trained **XGBoost model** to predict the probability of a product shortage for each order line. It then generates a prioritized list of actions for the dispatch team to proactively manage customer expectations.")
st.subheader("Model Performance (on Test Data)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Recall (Shortages Caught)", "98.9%")
col2.metric("Precision (Alarm Accuracy)", "73.8%")
col3.metric("F1-Score (Balanced Metric)", "0.845")
col4.metric("ROC AUC", "0.9986")
st.markdown("---")

# --- Live Single-Order Predictor ---
st.header("⚡ Live Single-Order Predictor")
with st.form("custom_order_form"):
    st.write("Enter the details of a single order line to get an instant shortage prediction.")
    customers = sorted(historical_data['customer_number'].unique().to_list())
    products = sorted(historical_data['product_code'].unique().to_list())
    plants = sorted(historical_data['plant'].unique().drop_nulls().to_list())
    storage_locs = sorted(historical_data['storage_location'].unique().drop_nulls().to_list())
    c1, c2, c3 = st.columns(3)
    with c1:
        customer_num = st.selectbox("Customer Number", customers, index=customers.index(33867) if 33867 in customers else 0)
        product_code = st.selectbox("Product Code", products, index=products.index("400070") if "400070" in products else 0)
    with c2:
        plant_num = st.selectbox("Fulfillment Plant", plants, index=plants.index(30588) if 30588 in plants else 0)
        storage_loc = st.selectbox("Storage Location", storage_locs, index=storage_locs.index(2012) if 2012 in storage_locs else 0)
    with c3:
        order_qty = st.number_input("Order Quantity", min_value=1.0, value=1.0, step=1.0)
        submitted = st.form_submit_button("🔮 Predict Shortage Risk")

if submitted:
    with st.spinner("Analyzing order..."):
        custom_order_df = pl.DataFrame({'order_number': [999999], 'order_created_date': [date.today()], 'customer_number': [customer_num], 'product_code': [str(product_code)], 'order_qty': [order_qty], 'plant': [plant_num], 'storage_location': [storage_loc]})
        prediction_result = predictor.predict(custom_order_df)
        prob = prediction_result['shortage_probability'][0]
        st.subheader("Prediction Result")
        pred_col1, pred_col2 = st.columns([1,2])
        with pred_col1:
            st.metric("Predicted Shortage Probability", f"{prob:.1%}")
            st.progress(prob)
            if prob >= Config.HIGH_RISK_THRESHOLD: st.error("High risk of shortage detected.")
            else: st.success("Low risk of shortage.")
        with pred_col2:
            st.code(predictor.explain_prediction_as_string(prediction_result))
st.markdown("---")

# --- Main Dispatch Simulation ---
st.header("📦 Daily Dispatch Simulation")
if st.button(f"▶️ Run Smart Dispatcher on {len(sample_orders)} New Orders"):
    with st.spinner("Analyzing orders and predicting shortages..."):
        predictions, actions = run_smart_dispatcher(predictor, sub_engine, sample_orders)
        st.session_state.predictions = predictions
        st.session_state.actions = actions
        st.session_state.dispatcher_run = True
        st.session_state.notification_details = {}

if st.session_state.dispatcher_run and st.session_state.predictions is not None:
    st.subheader("📈 Dispatch Summary")
    preds = st.session_state.predictions
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔴 Critical Risk", preds.filter(pl.col('risk_level') == '🔴 CRITICAL').height)
    c2.metric("🟡 High Risk", preds.filter(pl.col('risk_level') == '🟡 HIGH').height)
    c3.metric("🟠 Medium Risk", preds.filter(pl.col('risk_level') == '🟠 MEDIUM').height)
    c4.metric("🟢 Low Risk", preds.filter(pl.col('risk_level') == '🟢 LOW').height)
    
    actions_df = st.session_state.actions
    if actions_df is not None and actions_df.height > 0:
        st.info(f"**Action Required on {actions_df.height} orders.** These are prioritized below.")
        st.subheader("📋 Priority Action Items")
        
        actions_df_pd = actions_df.to_pandas()
        actions_df_pd['Generate AI Call'] = False
        actions_df_pd['Status'] = actions_df_pd['order_number'].apply(lambda x: "AI Call Generated ✅" if x in st.session_state.notification_details else "Pending")
        
        edited_df = st.data_editor(actions_df_pd, column_config={"Generate AI Call": st.column_config.CheckboxColumn(help="Generate an AI voice call script for this customer.", default=False), "shortage_probability": st.column_config.ProgressColumn("Probability", format="%.1f%%", min_value=0, max_value=1)}, disabled=["order_number", "risk_level", "product_code", "customer_number", "order_qty", "shortage_probability", "recommended_action", "substitutes", "Status"], hide_index=True, use_container_width=True)

        triggered_order = edited_df[edited_df["Generate AI Call"]]
        if not triggered_order.empty:
            order_details = triggered_order.iloc[0].to_dict()
            order_num = order_details['order_number']
            
            if order_num not in st.session_state.notification_details:
                with st.spinner(f"Generating AI call for order {order_num}... (Step 1/2: Script)"):
                    script = generate_call_script(order_details, mock_mode=mock_apis)
                if script:
                    with st.spinner(f"Generating AI call for order {order_num}... (Step 2/2: Audio)"):
                        audio_path = generate_audio_from_script(script, order_num, mock_mode=mock_apis)
                    if audio_path is not None:
                        st.session_state.notification_details[order_num] = {'script': script, 'audio_path': audio_path, 'details': order_details}
                        st.rerun()

        if st.session_state.notification_details:
            st.subheader("📞 Generated AI Customer Calls")
            for order_num, details in st.session_state.notification_details.items():
                with st.expander(f"**Order #{order_num}** | Product: `{details['details']['product_code']}` | Customer: `{details['details']['customer_number']}`"):
                    st.markdown("**Generated Call Script:**")
                    st.info(details['script'])
                    st.markdown("**Generated Audio:**")
                    if mock_apis or not Path(details['audio_path']).exists():
                        st.warning("Mock mode is on or audio file not found. Audio player is disabled.")
                    else:
                        st.audio(details['audio_path'])

        st.markdown("---")
        st.header("🔍 Order Deep Dive & Explanation")
        order_options = [f"Order: {r['order_number']} | Product: {r['product_code']}" for r in actions_df.iter_rows(named=True)]
        selected_order_str = st.selectbox("Select a high-risk order to analyze:", order_options)
        if selected_order_str:
            selected_order_num = int(selected_order_str.split(" | ")[0].split(": ")[1])
            selected_prod_code = selected_order_str.split(" | ")[1].split(": ")[1]
            order_details = preds.filter((pl.col('order_number') == selected_order_num) & (pl.col('product_code') == selected_prod_code))
            if order_details.height > 0:
                prob = order_details['shortage_probability'][0]
                st.subheader(f"Analysis for Order {selected_order_num} / Product {selected_prod_code}")
                exp_col1, exp_col2 = st.columns([1, 2])
                with exp_col1:
                    st.metric("Shortage Probability", f"{prob:.1%}")
                    st.progress(prob)
                    st.write(f"**Customer:** {order_details['customer_number'][0]}")
                with exp_col2:
                    st.code(predictor.explain_prediction_as_string(order_details))
                st.subheader("🔄 Suggested Substitutes (Visual)")
                substitute_codes = sub_engine.get_substitutes(selected_prod_code)
                if substitute_codes:
                    cols = st.columns(len(substitute_codes))
                    for i, sub_code in enumerate(substitute_codes):
                        with cols[i]:
                            st.markdown(f"**Code:** `{sub_code}`")
                            image_url = MOCK_PRODUCT_IMAGES.get(sub_code, {}).get("url", f"https://placehold.co/150x150/9E9E9E/white?text={sub_code}")
                            st.image(image_url, caption=MOCK_PRODUCT_IMAGES.get(sub_code, {}).get("name", f"Product {sub_code}"))
                else:
                    st.warning("No direct substitutes found.")
    else:
        st.success("✅ No high-risk orders detected in this batch.")

# --- Post-Delivery AI Inspector ---
st.markdown("---")
st.header("📸 Post-Delivery AI Inspector (Multimodal Demo)")
st.write("Simulate a customer uploading a photo of a delivery issue. The AI will identify the product and check its current shortage risk.")
uploaded_file = st.file_uploader("Upload an image of a damaged or incorrect item", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img_bytes = uploaded_file.getvalue()
    col_img, col_result = st.columns([1, 2])
    with col_img:
        st.image(img_bytes, caption="Customer's Uploaded Image")

    if st.button("Analyze Image and Check Status"):
        with st.spinner("Step 1/2: AI is identifying the product in the image..."):
            identified_pid, debug_info = identify_product_in_image(img_bytes, mock_mode=mock_apis)
        
        if st.session_state.debug_mode:
            with st.expander("🔬 View Vision AI Debug Log"):
                st.text(debug_info)

        if identified_pid:
            st.success(f"✅ Product identified by Vision AI: **{MOCK_PRODUCT_IMAGES[identified_pid]['name']}** (ID: `{identified_pid}`)")
            with st.spinner(f"Step 2/2: Checking current shortage status for product {identified_pid}..."):
                most_common_customer = historical_data['customer_number'].mode()[0]
                most_common_plant = historical_data['plant'].mode()[0]
                most_common_storage = historical_data['storage_location'].mode()[0]
                dummy_order = pl.DataFrame({'order_number': [999998], 'order_created_date': [date.today()], 'customer_number': [most_common_customer], 'product_code': [identified_pid], 'order_qty': [1.0], 'plant': [most_common_plant], 'storage_location': [most_common_storage]})
                status_prediction = predictor.predict(dummy_order)
                shortage_prob = status_prediction['shortage_probability'][0]
            
            with col_result:
                st.subheader("AI Analysis & Status Check")
                st.metric("Current Predicted Shortage Risk", f"{shortage_prob:.1%}")
                if shortage_prob >= Config.HIGH_RISK_THRESHOLD:
                    st.error("This product is currently experiencing a high risk of shortages across orders.")
                else:
                    st.success("This product has a low shortage risk at the moment.")
        else:
            st.error("❌ AI could not confidently identify a known product. Check the debug log (if enabled) for details.")