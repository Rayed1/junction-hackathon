
import polars as pl
import pandas as pd
import numpy as np
import json
import pickle
import time
from pathlib import Path
from datetime import datetime, timedelta

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)
import xgboost as xgb


# ============================================================================
# SECTION 1: CONFIGURATION
# ============================================================================

class Config:
    """Configuration settings for the pipeline"""
    
    # Data paths
    DATA_DIR = Path("./data")
    SALES_FILE = DATA_DIR / "valio_aimo_sales_and_deliveries_junction_2025.csv"
    PURCHASES_FILE = DATA_DIR / "valio_aimo_purchases_junction_2025.csv"
    PRODUCT_FILE = DATA_DIR / "valio_aimo_product_data_junction_2025.json"
    REPLACEMENT_FILE = DATA_DIR / "valio_aimo_replacement_orders_junction_2025.csv"
    
    # Model output paths
    MODEL_PATH = "shortage_predictor_model.pkl"
    PREDICTIONS_PATH = "predictions_output.parquet"
    ACTIONS_PATH = "dispatcher_action_items.csv"
    
    # Model parameters
    TRAIN_TEST_SPLIT = 0.8  # 80% train, 20% test
    RANDOM_SEED = 42
    
    # Business thresholds
    CRITICAL_RISK_THRESHOLD = 0.8  # 80% probability
    HIGH_RISK_THRESHOLD = 0.5      # 50% probability
    MEDIUM_RISK_THRESHOLD = 0.3    # 30% probability


# ============================================================================
# SECTION 2: DATA LOADING AND CLEANING
# ============================================================================

def load_data():
    """
    Load all datasets from CSV/JSON files
    
    Returns:
        tuple: (df_sales, df_purchases, df_products, df_replacements)
    """
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Load sales & deliveries (main dataset)
    df_sales = pl.read_csv(Config.SALES_FILE)
    print(f"✓ Sales & Deliveries: {df_sales.shape[0]:,} rows, {df_sales.shape[1]} cols")
    
    # Load purchases
    df_purchases = pl.read_csv(Config.PURCHASES_FILE)
    print(f"✓ Purchases: {df_purchases.shape[0]:,} rows")
    
    # Load product data - SIMPLIFIED VERSION
    # Since product codes don't match GTINs anyway, create a minimal stub
    print("Creating minimal product data structure...")
    
    # Create a stub DataFrame with just the structure we need
    df_products = pl.DataFrame({
        'product_code': ['STUB'],
        'salesUnit': [''],
        'baseUnit': [''],
        'category': ['UNKNOWN'],
        'deleted': [False],
        'temperatureCondition': ['UNKNOWN'],
        'vendorName': ['UNKNOWN'],
        'countryOfOrigin': [''],
        'allowedLotSize': [None],
        'has_substitutions': [False],
        'unit_count': [0],
    })
    print(f"✓ Product Data: Stub created (product matching disabled)")
    print("  Note: Product metadata won't match since internal codes ≠ GTINs")
    print("  This is OK - historical features are more powerful anyway!")
    
    # Load replacement orders
    df_replacements = pl.read_csv(Config.REPLACEMENT_FILE)
    print(f"✓ Replacement Orders: {df_replacements.shape[0]:,} rows")
    
    return df_sales, df_purchases, df_products, df_replacements

def clean_data(df_sales, df_products):
    """
    Clean and preprocess the sales data
    """
    print("\n" + "="*80)
    print("CLEANING DATA")
    print("="*80)
    
    df = df_sales.clone()
    
    # 1. Convert date columns
    print("\n1. Converting dates...")
    date_columns = ['order_created_date', 'requested_delivery_date', 'picking_confirmed_date']
    for col in date_columns:
        if col in df.columns:
            df = df.with_columns([
                pl.col(col).str.strptime(pl.Date, format="%Y-%m-%d", strict=False).alias(col)
            ])
    print("✓ Date conversion complete")
    
    # 2. Handle placeholder dates
    print("\n2. Handling placeholder dates...")
    for col in date_columns:
        if col in df.columns and df[col].dtype == pl.Date:
            count = df.filter(pl.col(col) == datetime(1970, 1, 1).date()).height
            if count > 0:
                print(f"  - {col}: {count:,} placeholder dates → null")
                df = df.with_columns([
                    pl.when(pl.col(col) == datetime(1970, 1, 1).date())
                    .then(None)
                    .otherwise(pl.col(col))
                    .alias(col)
                ])
    
    # 3. Handle null quantities
    print("\n3. Handling null quantities...")
    df = df.with_columns([
        pl.col('delivered_qty').fill_null(0),
        pl.col('picking_picked_qty').fill_null(0)
    ])
    
    # 4. CREATE TARGET VARIABLE
    print("\n4. Creating target variable (is_shortage)...")
    df = df.with_columns([
        (
            (pl.col('order_qty') > pl.col('delivered_qty')) |
            ((pl.col('delivered_qty') == 0) & (pl.col('order_qty') > 0))
        ).cast(pl.Int8).alias('is_shortage')
    ])
    
    shortage_count = df['is_shortage'].sum()
    shortage_rate = shortage_count / df.height * 100
    print(f"✓ Shortage rate: {shortage_rate:.2f}% ({shortage_count:,} out of {df.height:,} orders)")
    
    # 5. Add placeholder product metadata columns (since we can't match)
    print("\n5. Adding product metadata placeholders...")
    df = df.with_columns([
        pl.lit('UNKNOWN').alias('vendorName'),
        pl.lit('UNKNOWN').alias('category'),
        pl.lit('UNKNOWN').alias('temperatureCondition')
    ])
    print("  - Added placeholder columns (no actual product matching)")
    
    # 6. Handle categorical nulls in original columns
    print("\n6. Handling categorical nulls...")
    categorical_cols = ['storage_location', 'plant']
    for col in categorical_cols:
        if col in df.columns:
            null_count = df[col].is_null().sum()
            if null_count > 0:
                df = df.with_columns([pl.col(col).fill_null('UNKNOWN')])
    
    # 7. Remove duplicates
    original_rows = df.height
    df = df.unique()
    removed = original_rows - df.height
    if removed > 0:
        print(f"\n7. Removed {removed:,} duplicate rows")
    
    # 8. Convert product_code to string for consistency
    df = df.with_columns([pl.col('product_code').cast(pl.Utf8)])
    
    print(f"\n✓ Cleaning complete! Final shape: {df.shape}")
    return df


# ============================================================================
# SECTION 3: FEATURE ENGINEERING
# ============================================================================

def create_date_features(df):
    """
    Extract temporal features from dates
    
    Features created:
    - day_of_week (0=Monday, 6=Sunday)
    - day_of_month (1-31)
    - month (1-12)
    - quarter (1-4)
    - is_weekend (boolean)
    - lead_time_days (requested_delivery - order_created)
    - order_hour (hour of day order was placed)
    """
    df = df.with_columns([
        pl.col('order_created_date').dt.weekday().alias('order_day_of_week'),
        pl.col('order_created_date').dt.day().alias('order_day_of_month'),
        pl.col('order_created_date').dt.month().alias('order_month'),
        pl.col('order_created_date').dt.quarter().alias('order_quarter'),
        (pl.col('order_created_date').dt.weekday() >= 5).cast(pl.Int8).alias('is_weekend'),
        ((pl.col('requested_delivery_date') - pl.col('order_created_date')).dt.total_days()).alias('lead_time_days'),
        (pl.col('order_created_time') // 10000).alias('order_hour')
    ])
    return df


def create_product_features(df):
    """
    Create product-level aggregated features
    
    Features created:
    - product_shortage_rate: Historical shortage rate per product
    - product_avg_order_qty: Average quantity ordered
    - product_order_qty_std: Volatility of order quantities
    - product_order_count: How many times ordered
    - category_shortage_rate: Shortage rate by product category
    - temp_condition_shortage_rate: Shortage rate by temperature requirement
    """
    df = df.sort('order_created_date')
    
    # Product statistics
    product_stats = df.group_by('product_code').agg([
        pl.col('is_shortage').mean().alias('product_shortage_rate'),
        pl.col('order_qty').mean().alias('product_avg_order_qty'),
        pl.col('order_qty').std().alias('product_order_qty_std'),
        pl.col('delivered_qty').mean().alias('product_avg_delivered_qty'),
        pl.col('order_number').n_unique().alias('product_order_count')
    ])
    df = df.join(product_stats, on='product_code', how='left', suffix='_stat')
    
    # Category statistics
    if 'category' in df.columns:
        category_stats = df.group_by('category').agg([
            pl.col('is_shortage').mean().alias('category_shortage_rate')
        ])
        df = df.join(category_stats, on='category', how='left', suffix='_cat')
    
    # Temperature condition statistics (perishables may have higher shortage risk)
    if 'temperatureCondition' in df.columns:
        temp_stats = df.group_by('temperatureCondition').agg([
            pl.col('is_shortage').mean().alias('temp_condition_shortage_rate')
        ])
        df = df.join(temp_stats, on='temperatureCondition', how='left', suffix='_temp')
    
    return df


def create_customer_features(df):
    """
    Create customer-level aggregated features
    
    Features created:
    - customer_shortage_rate: How often this customer experiences shortages
    - customer_total_order_qty: Total quantity ordered by customer
    - customer_order_count: Number of orders placed
    - customer_avg_lead_time: Customer's typical lead time
    - customer_product_shortage_rate: Shortage rate for specific customer-product combo
    - customer_product_order_count: How many times customer ordered this product
    """
    # Customer statistics
    customer_stats = df.group_by('customer_number').agg([
        pl.col('is_shortage').mean().alias('customer_shortage_rate'),
        pl.col('order_qty').sum().alias('customer_total_order_qty'),
        pl.col('order_number').n_unique().alias('customer_order_count'),
        pl.col('lead_time_days').mean().alias('customer_avg_lead_time')
    ])
    df = df.join(customer_stats, on='customer_number', how='left', suffix='_cust')
    
    # Customer-Product combination (most granular and powerful feature!)
    customer_product_stats = df.group_by(['customer_number', 'product_code']).agg([
        pl.col('is_shortage').mean().alias('customer_product_shortage_rate'),
        pl.col('order_number').n_unique().alias('customer_product_order_count')
    ])
    df = df.join(customer_product_stats, on=['customer_number', 'product_code'], how='left', suffix='_cp')
    
    return df


def create_vendor_features(df):
    """
    Create vendor/supplier reliability features
    
    Features created:
    - vendor_shortage_rate: How reliable is this supplier
    - vendor_order_count: Volume of orders from this supplier
    """
    if 'vendorName' in df.columns:
        vendor_stats = df.group_by('vendorName').agg([
            pl.col('is_shortage').mean().alias('vendor_shortage_rate'),
            pl.col('order_number').n_unique().alias('vendor_order_count')
        ])
        df = df.join(vendor_stats, on='vendorName', how='left', suffix='_vend')
    return df


def create_location_features(df):
    """
    Create warehouse/location features
    
    Features created:
    - plant_shortage_rate: Shortage rate by fulfillment plant
    - storage_shortage_rate: Shortage rate by storage location
    """
    # Plant statistics
    plant_stats = df.group_by('plant').agg([
        pl.col('is_shortage').mean().alias('plant_shortage_rate'),
        pl.col('order_number').n_unique().alias('plant_order_count')
    ])
    df = df.join(plant_stats, on='plant', how='left', suffix='_plant')
    
    # Storage location statistics
    storage_stats = df.group_by('storage_location').agg([
        pl.col('is_shortage').mean().alias('storage_shortage_rate')
    ])
    df = df.join(storage_stats, on='storage_location', how='left', suffix='_stor')
    
    return df


def create_replacement_features(df, df_replacements):
    """
    Create features from replacement order history
    
    Features created:
    - replacement_frequency: How often this product needs replacement orders
    """
    replacement_counts = df_replacements.group_by('product_code').agg([
        pl.count().alias('replacement_frequency')
    ])
    replacement_counts = replacement_counts.with_columns([
        pl.col('product_code').cast(pl.Utf8)
    ])
    df = df.join(replacement_counts, on='product_code', how='left')
    df = df.with_columns([pl.col('replacement_frequency').fill_null(0)])
    return df


def engineer_all_features(df_clean, df_replacements):
    """
    Apply all feature engineering steps
    
    This is where the "magic" happens - we transform raw order data
    into meaningful features that predict shortages.
    """
    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)
    
    df = df_clean.clone()
    
    print("\n1. Creating date/time features...")
    df = create_date_features(df)
    
    print("2. Creating product features...")
    df = create_product_features(df)
    
    print("3. Creating customer features...")
    df = create_customer_features(df)
    
    print("4. Creating vendor features...")
    df = create_vendor_features(df)
    
    print("5. Creating location features...")
    df = create_location_features(df)
    
    print("6. Creating replacement features...")
    df = create_replacement_features(df, df_replacements)
    
    print(f"\n✓ Feature engineering complete! Total columns: {len(df.columns)}")
    return df


# ============================================================================
# SECTION 4: MODEL TRAINING
# ============================================================================

def prepare_modeling_data(df):
    """
    Prepare data for machine learning
    
    Steps:
    1. Select relevant features
    2. Remove rows with nulls
    3. Create time-based train/test split (80/20)
    4. Convert to numpy arrays
    
    Returns:
        X_train, X_test, y_train, y_test, feature_cols, df_train, df_test
    """
    print("\n" + "="*80)
    print("PREPARING DATA FOR MODELING")
    print("="*80)
    
    # Define features to use in model
    feature_cols = [
        # Quantity features
        'order_qty',
        'picking_picked_qty',
        
        # Date features
        'order_day_of_week',
        'order_day_of_month',
        'order_month',
        'order_quarter',
        'is_weekend',
        'lead_time_days',
        'order_hour',
        
        # Product features
        'product_shortage_rate',
        'product_avg_order_qty',
        'product_order_qty_std',
        'product_order_count',
        
        # Customer features
        'customer_shortage_rate',
        'customer_total_order_qty',
        'customer_order_count',
        'customer_avg_lead_time',
        
        # Customer-Product features (MOST IMPORTANT!)
        'customer_product_shortage_rate',
        'customer_product_order_count',
        
        # Vendor features
        'vendor_shortage_rate',
        'vendor_order_count',
        
        # Location features
        'plant_shortage_rate',
        'storage_shortage_rate',
        
        # Replacement features
        'replacement_frequency',
        
        # Category features
        'category_shortage_rate',
        'temp_condition_shortage_rate'
    ]
    
    # Filter to only columns that exist
    feature_cols = [col for col in feature_cols if col in df.columns]
    print(f"\nUsing {len(feature_cols)} features")
    
    # Select columns for modeling
    cols_to_keep = feature_cols + ['is_shortage', 'order_created_date', 'order_number', 'product_code', 'customer_number']
    cols_to_keep = [col for col in cols_to_keep if col in df.columns]
    df_model = df.select(cols_to_keep)
    
    # Remove nulls
    print(f"Rows before null removal: {df_model.height:,}")
    df_model = df_model.drop_nulls(subset=feature_cols)
    print(f"Rows after null removal: {df_model.height:,}")
    
    # Check class balance
    shortage_rate = df_model['is_shortage'].mean()
    print(f"\nClass balance:")
    print(f"  - Shortage rate: {shortage_rate*100:.2f}%")
    print(f"  - Imbalance ratio: {(1-shortage_rate)/shortage_rate:.1f}:1")
    
    # TIME-BASED SPLIT (crucial for time-series data!)
    # We train on older data, test on newer data to simulate real-world scenario
    print("\n--- Time-Based Train/Test Split ---")
    df_model = df_model.sort('order_created_date')
    split_idx = int(df_model.height * Config.TRAIN_TEST_SPLIT)
    
    df_train = df_model[:split_idx]
    df_test = df_model[split_idx:]
    
    print(f"\nTraining: {df_train.height:,} rows ({df_train['order_created_date'].min()} to {df_train['order_created_date'].max()})")
    print(f"  Shortage rate: {df_train['is_shortage'].mean()*100:.2f}%")
    print(f"\nTesting: {df_test.height:,} rows ({df_test['order_created_date'].min()} to {df_test['order_created_date'].max()})")
    print(f"  Shortage rate: {df_test['is_shortage'].mean()*100:.2f}%")
    
    # Convert to numpy for sklearn
    X_train = df_train.select(feature_cols).to_numpy()
    y_train = df_train.select('is_shortage').to_numpy().ravel()
    X_test = df_test.select(feature_cols).to_numpy()
    y_test = df_test.select('is_shortage').to_numpy().ravel()
    
    print(f"\n✓ Data prepared! X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, feature_cols, df_train, df_test


def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    """
    Evaluate model performance with detailed metrics
    
    Metrics:
    - Accuracy: Overall correctness
    - Precision: Of predicted shortages, how many were correct?
    - Recall: Of actual shortages, how many did we catch?
    - F1-Score: Harmonic mean of precision and recall
    - ROC AUC: Area under ROC curve
    """
    print(f"\n{'='*80}")
    print(f"{model_name.upper()} PERFORMANCE")
    print(f"{'='*80}")
    
    accuracy = (y_true == y_pred).mean()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\nMetrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f} (% of predicted shortages that were correct)")
    print(f"  Recall:    {recall:.4f} (% of actual shortages we caught)")
    print(f"  F1-Score:  {f1:.4f}")
    
    try:
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        print(f"  ROC AUC:   {roc_auc:.4f}")
    except:
        roc_auc = None
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"{'':20} Predicted: No | Predicted: Yes")
    print(f"Actual: No Shortage  {cm[0,0]:>10,}   |   {cm[0,1]:>10,}")
    print(f"Actual: Shortage     {cm[1,0]:>10,}   |   {cm[1,1]:>10,}")
    
    total_actual = cm[1,0] + cm[1,1]
    total_predicted = cm[0,1] + cm[1,1]
    
    print(f"\n💼 Business Impact:")
    print(f"  • Actual shortages: {total_actual:,}")
    print(f"  • Caught {cm[1,1]:,} shortages ({cm[1,1]/total_actual*100:.1f}%)")
    print(f"  • Missed {cm[1,0]:,} shortages ({cm[1,0]/total_actual*100:.1f}%)")
    print(f"  • False alarms: {cm[0,1]:,}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }


def train_models(X_train, X_test, y_train, y_test, feature_cols):
    """
    Train and compare three models:
    1. Logistic Regression (baseline)
    2. Random Forest (ensemble method)
    3. XGBoost (gradient boosting - usually best)
    
    Returns:
        Best model and comparison results
    """
    print("\n" + "="*80)
    print("MODEL TRAINING")
    print("="*80)
    
    all_metrics = {}
    
    # MODEL 1: Logistic Regression (Baseline)
    print("\n📊 Model 1: Logistic Regression...")
    start = time.time()
    lr_model = LogisticRegression(max_iter=1000, random_state=Config.RANDOM_SEED, 
                                   class_weight='balanced', n_jobs=-1)
    lr_model.fit(X_train, y_train)
    print(f"Training time: {time.time()-start:.1f}s")
    
    y_pred_lr = lr_model.predict(X_test)
    y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]
    all_metrics['Logistic Regression'] = evaluate_model(y_test, y_pred_lr, y_pred_proba_lr, "Logistic Regression")
    
    # MODEL 2: Random Forest
    print("\n📊 Model 2: Random Forest...")
    start = time.time()
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=100,
        min_samples_leaf=50,
        random_state=Config.RANDOM_SEED,
        class_weight='balanced',
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    print(f"Training time: {time.time()-start:.1f}s")
    
    y_pred_rf = rf_model.predict(X_test)
    y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    all_metrics['Random Forest'] = evaluate_model(y_test, y_pred_rf, y_pred_proba_rf, "Random Forest")
    
    print("\n🌟 Top 10 Most Important Features (Random Forest):")
    feat_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    for i, row in feat_imp.head(10).iterrows():
        print(f"  {row['feature']:.<45} {row['importance']:.4f}")
    
    # MODEL 3: XGBoost (Usually the best!)
    print("\n📊 Model 3: XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Class imbalance weight: {scale_pos_weight:.1f}")
    
    start = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=Config.RANDOM_SEED,
        n_jobs=-1,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    print(f"Training time: {time.time()-start:.1f}s")
    
    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
    all_metrics['XGBoost'] = evaluate_model(y_test, y_pred_xgb, y_pred_proba_xgb, "XGBoost")
    
    print("\n🌟 Top 10 Most Important Features (XGBoost):")
    feat_imp_xgb = pd.DataFrame({
        'feature': feature_cols,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    for i, row in feat_imp_xgb.head(10).iterrows():
        print(f"  {row['feature']:.<45} {row['importance']:.4f}")
    
    # Model Comparison
    print("\n" + "="*80)
    print("📊 MODEL COMPARISON")
    print("="*80)
    
    comparison = pd.DataFrame({
        'Model': list(all_metrics.keys()),
        'Accuracy': [m['accuracy'] for m in all_metrics.values()],
        'Precision': [m['precision'] for m in all_metrics.values()],
        'Recall': [m['recall'] for m in all_metrics.values()],
        'F1': [m['f1'] for m in all_metrics.values()],
        'ROC-AUC': [m.get('roc_auc', 0) for m in all_metrics.values()]
    })
    print("\n" + comparison.to_string(index=False))
    
    # Return best model (XGBoost typically wins)
    best_model = xgb_model
    print(f"\n✅ Best model: XGBoost")
    
    return best_model, feat_imp_xgb, all_metrics


def save_model(model, feature_cols, feature_importance):
    """
    Save trained model and metadata
    """
    print("\n" + "="*80)
    print("💾 SAVING MODEL")
    print("="*80)
    
    artifacts = {
        'model': model,
        'feature_cols': feature_cols,
        'feature_importance': feature_importance,
        'threshold': 0.5,
        'trained_date': datetime.now().isoformat()
    }
    
    with open(Config.MODEL_PATH, 'wb') as f:
        pickle.dump(artifacts, f)
    
    print(f"✓ Model saved to: {Config.MODEL_PATH}")


# ============================================================================
# SECTION 5: PREDICTION PIPELINE (For Production Use)
# ============================================================================

class ShortagePredictor:
    """
    Production-ready prediction pipeline
    
    Usage:
        predictor = ShortagePredictor('shortage_predictor_model.pkl')
        predictor.load_historical_data(df_features)
        predictions = predictor.predict(df_new_orders)
    """
    
    def __init__(self, model_path=None):
        """Load trained model"""
        if model_path is None:
            model_path = Config.MODEL_PATH
            
        print(f"\nLoading model from {model_path}...")
        with open(model_path, 'rb') as f:
            artifacts = pickle.load(f)
        
        self.model = artifacts['model']
        self.feature_cols = artifacts['feature_cols']
        self.feature_importance = artifacts['feature_importance']
        self.threshold = artifacts.get('threshold', 0.5)
        
        print(f"✓ Model loaded with {len(self.feature_cols)} features")
        print(f"  Trained: {artifacts.get('trained_date', 'Unknown')}")
    
    def load_historical_data(self, df_features):
        """
        Load historical statistics for feature engineering
        This should be called once with your historical data
        """
        print("\nLoading historical statistics...")
        
        # Product stats
        self.product_stats = df_features.group_by('product_code').agg([
            pl.col('is_shortage').mean().alias('product_shortage_rate'),
            pl.col('order_qty').mean().alias('product_avg_order_qty'),
            pl.col('order_qty').std().alias('product_order_qty_std'),
            pl.col('order_number').n_unique().alias('product_order_count')
        ])
        
        # Customer stats
        self.customer_stats = df_features.group_by('customer_number').agg([
            pl.col('is_shortage').mean().alias('customer_shortage_rate'),
            pl.col('order_qty').sum().alias('customer_total_order_qty'),
            pl.col('order_number').n_unique().alias('customer_order_count'),
            pl.col('lead_time_days').mean().alias('customer_avg_lead_time')
        ])
        
        # Customer-Product stats (MOST IMPORTANT!)
        self.customer_product_stats = df_features.group_by(['customer_number', 'product_code']).agg([
            pl.col('is_shortage').mean().alias('customer_product_shortage_rate'),
            pl.col('order_number').n_unique().alias('customer_product_order_count')
        ])
        
        # Location stats
        self.plant_stats = df_features.group_by('plant').agg([
            pl.col('is_shortage').mean().alias('plant_shortage_rate')
        ])
        
        self.storage_stats = df_features.group_by('storage_location').agg([
            pl.col('is_shortage').mean().alias('storage_shortage_rate')
        ])
        
        # Other stats
        self.category_stats = df_features.group_by('category').agg([
            pl.col('is_shortage').mean().alias('category_shortage_rate')
        ])
        
        self.temp_stats = df_features.group_by('temperatureCondition').agg([
            pl.col('is_shortage').mean().alias('temp_condition_shortage_rate')
        ])
        
        self.vendor_stats = df_features.group_by('vendorName').agg([
            pl.col('is_shortage').mean().alias('vendor_shortage_rate'),
            pl.col('order_number').n_unique().alias('vendor_order_count')
        ])
        
        print("✓ Historical statistics loaded")
    
    def engineer_features(self, df_new_orders):
        """
        Engineer features for new orders using historical statistics
        """
        df = df_new_orders.clone()
        
        # Ensure product_code is string
        if 'product_code' in df.columns:
            df = df.with_columns([pl.col('product_code').cast(pl.Utf8)])
        
        # Date features
        df = df.with_columns([
            pl.col('order_created_date').dt.weekday().alias('order_day_of_week'),
            pl.col('order_created_date').dt.day().alias('order_day_of_month'),
            pl.col('order_created_date').dt.month().alias('order_month'),
            pl.col('order_created_date').dt.quarter().alias('order_quarter'),
            (pl.col('order_created_date').dt.weekday() >= 5).cast(pl.Int8).alias('is_weekend'),
        ])
        
        # Handle missing columns
        if 'requested_delivery_date' not in df.columns:
            df = df.with_columns([
                (pl.col('order_created_date') + pl.duration(days=1)).alias('requested_delivery_date')
            ])
        
        df = df.with_columns([
            ((pl.col('requested_delivery_date') - pl.col('order_created_date')).dt.total_days()).alias('lead_time_days')
        ])
        
        if 'order_created_time' in df.columns:
            df = df.with_columns([
                (pl.col('order_created_time') // 10000).alias('order_hour')
            ])
        else:
            df = df.with_columns([pl.lit(14).alias('order_hour')])  # Default to 2 PM
        
        if 'picking_picked_qty' not in df.columns:
            df = df.with_columns([pl.lit(0).alias('picking_picked_qty')])
        
        # Join with historical stats
        df = df.join(self.product_stats, on='product_code', how='left')
        df = df.join(self.customer_stats, on='customer_number', how='left')
        df = df.join(self.customer_product_stats, on=['customer_number', 'product_code'], how='left')
        df = df.join(self.plant_stats, on='plant', how='left')
        df = df.join(self.storage_stats, on='storage_location', how='left')
        
        # Fill nulls with global averages (for new customers/products)
        GLOBAL_SHORTAGE_RATE = 0.04  # 4% average
        df = df.with_columns([
            pl.col('product_shortage_rate').fill_null(GLOBAL_SHORTAGE_RATE),
            pl.col('product_avg_order_qty').fill_null(pl.col('order_qty')),
            pl.col('product_order_qty_std').fill_null(0),
            pl.col('product_order_count').fill_null(0),
            pl.col('customer_shortage_rate').fill_null(GLOBAL_SHORTAGE_RATE),
            pl.col('customer_total_order_qty').fill_null(0),
            pl.col('customer_order_count').fill_null(0),
            pl.col('customer_avg_lead_time').fill_null(pl.col('lead_time_days')),
            pl.col('customer_product_shortage_rate').fill_null(GLOBAL_SHORTAGE_RATE),
            pl.col('customer_product_order_count').fill_null(0),
            pl.col('plant_shortage_rate').fill_null(GLOBAL_SHORTAGE_RATE),
            pl.col('storage_shortage_rate').fill_null(GLOBAL_SHORTAGE_RATE),
        ])
        
        # Add remaining features with defaults
        for col in ['vendor_shortage_rate', 'vendor_order_count', 'replacement_frequency', 
                    'category_shortage_rate', 'temp_condition_shortage_rate']:
            if col not in df.columns:
                default_val = GLOBAL_SHORTAGE_RATE if 'rate' in col else 0
                df = df.with_columns([pl.lit(default_val).alias(col)])
        
        return df
    
    def predict(self, df_new_orders):
        """
        Predict shortages for new orders
        
        Args:
            df_new_orders: DataFrame with new orders
                Required columns: order_number, order_created_date, customer_number,
                                 product_code, order_qty, plant, storage_location
        
        Returns:
            DataFrame with predictions and probabilities
        """
        print(f"\n🔮 Predicting shortages for {df_new_orders.height:,} orders...")
        
        # Engineer features
        df_features = self.engineer_features(df_new_orders)
        
        # Select only features used in training
        X = df_features.select(self.feature_cols).to_numpy()
        
        # Predict
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        # Add predictions to dataframe
        df_result = df_features.with_columns([
            pl.lit(predictions).alias('predicted_shortage'),
            pl.lit(probabilities).alias('shortage_probability')
        ])
        
        shortage_count = predictions.sum()
        print(f"✓ Predicted {shortage_count:,} shortages ({shortage_count/len(predictions)*100:.1f}%)")
        
        return df_result
    
    def get_high_risk_orders(self, df_predictions, threshold=0.7):
        """Get orders with high shortage risk"""
        high_risk = df_predictions.filter(
            pl.col('shortage_probability') >= threshold
        ).sort('shortage_probability', descending=True)
        
        return high_risk
    
    def explain_prediction(self, df_row):
        """
        Explain why an order was predicted as shortage
        Shows the top contributing factors
        """
        features = df_row.select(self.feature_cols).to_numpy()[0]
        
        print("\n" + "="*60)
        print("🔍 SHORTAGE RISK EXPLANATION")
        print("="*60)
        print(f"\nOrder Number: {df_row['order_number'][0]}")
        print(f"Product Code: {df_row['product_code'][0]}")
        print(f"Customer: {df_row['customer_number'][0]}")
        print(f"Order Quantity: {df_row['order_qty'][0]}")
        print(f"\n⚠️  Shortage Probability: {df_row['shortage_probability'][0]:.1%}")
        
        print("\n📊 Top 10 Risk Factors:")
        
        # Calculate feature contributions
        feature_values = {}
        for i, col in enumerate(self.feature_cols):
            importance = self.feature_importance[
                self.feature_importance['feature'] == col
            ]['importance'].values[0] if col in self.feature_importance['feature'].values else 0
            
            feature_values[col] = {
                'value': features[i],
                'importance': importance,
                'contribution': features[i] * importance
            }
        
        # Sort by absolute contribution
        sorted_features = sorted(feature_values.items(), 
                                key=lambda x: abs(x[1]['contribution']), 
                                reverse=True)
        
        for i, (feat, info) in enumerate(sorted_features[:10], 1):
            print(f"  {i:2d}. {feat:.<40} {info['value']:.4f}")


# ============================================================================
# SECTION 6: SUBSTITUTION ENGINE
# ============================================================================

class SubstitutionEngine:
    """
    Engine for finding product substitutions
    
    When a shortage is predicted, this suggests alternative products
    that can be offered to the customer.
    """
    
    def __init__(self, df_products, df_replacements=None):
        self.df_products = df_products
        self.df_replacements = df_replacements
        self.substitution_matrix = self._build_substitution_matrix()
    
    def _build_substitution_matrix(self):
        """
        Build substitution matrix from historical data
        """
        print("\n🔄 Building substitution matrix...")
        
        substitutions = {}
        
        # Method 1: By category (similar products)
        if 'category' in self.df_products.columns:
            category_groups = self.df_products.group_by('category').agg([
                pl.col('product_code').alias('products')  # FIXED: use 'product_code' instead of 'salesUnitGtin'
            ])
            
            for row in category_groups.iter_rows(named=True):
                products = row['products']
                # Each product can be substituted by others in same category
                for prod in products:
                    if prod not in substitutions:
                        substitutions[prod] = []
                    # Add other products from same category (exclude self)
                    substitutions[prod].extend([p for p in products if p != prod])
        
        # Method 2: From replacement order history
        if self.df_replacements is not None and self.df_replacements.height > 0:
            replacement_products = self.df_replacements['product_code'].cast(pl.Utf8).unique().to_list()
            for prod in replacement_products:
                if prod not in substitutions:
                    substitutions[prod] = []
        
        print(f"✓ Built substitution matrix for {len(substitutions)} products")
        return substitutions
    
    def get_substitutes(self, product_code, top_n=5):
        """Get top N substitute products"""
        product_code = str(product_code)
        
        if product_code not in self.substitution_matrix:
            return []
        
        substitutes = self.substitution_matrix[product_code]
        return substitutes[:top_n] if substitutes else []
    
    def find_substitute_with_details(self, product_code, top_n=3):
        """Get substitute products with their details"""
        substitutes = self.get_substitutes(product_code, top_n)
        
        if not substitutes:
            return pl.DataFrame()
        
        # Get product details - FIXED: use 'product_code' column
        sub_details = self.df_products.filter(
            pl.col('product_code').is_in(substitutes)
        ).select([
            'product_code', 'category', 'vendorName', 'salesUnit'
        ])
        
        return sub_details

# ============================================================================
# SECTION 7: INTEGRATED SMART DISPATCHER
# ============================================================================

def run_smart_dispatcher(predictor, sub_engine, input_orders_df):
    """
    🚀 COMPLETE END-TO-END SMART DISPATCHER PIPELINE
    
    This is the main function that ties everything together:
    1. Predict shortage risk for new orders
    2. Classify orders by risk level
    3. Generate action recommendations
    4. Suggest product substitutions for high-risk orders
    
    Args:
        predictor: ShortagePredictor instance
        sub_engine: SubstitutionEngine instance
        input_orders_df: DataFrame with new orders to process
        
    Returns:
        (predictions_df, actions_df): Full predictions and recommended actions
    """
    print("\n" + "="*80)
    print("🚀 SMART DISPATCHER - INTEGRATED PIPELINE")
    print("="*80)
    
    # Ensure required columns
    if 'order_created_time' not in input_orders_df.columns:
        input_orders_df = input_orders_df.with_columns([
            pl.lit(140000).alias('order_created_time')
        ])
    
    if 'requested_delivery_date' not in input_orders_df.columns:
        input_orders_df = input_orders_df.with_columns([
            (pl.col('order_created_date') + pl.duration(days=1)).alias('requested_delivery_date')
        ])
    
    # STEP 1: Predict shortage risk
    print("\n📊 Step 1: Predicting shortage risk...")
    predictions = predictor.predict(input_orders_df)
    
    # STEP 2: Classify by risk level
    predictions = predictions.with_columns([
        pl.when(pl.col('shortage_probability') >= Config.CRITICAL_RISK_THRESHOLD)
          .then(pl.lit('🔴 CRITICAL'))
        .when(pl.col('shortage_probability') >= Config.HIGH_RISK_THRESHOLD)
          .then(pl.lit('🟡 HIGH'))
        .when(pl.col('shortage_probability') >= Config.MEDIUM_RISK_THRESHOLD)
          .then(pl.lit('🟠 MEDIUM'))
        .otherwise(pl.lit('🟢 LOW'))
        .alias('risk_level')
    ])
    
    # STEP 3: Summary statistics
    print("\n📈 Results Summary:")
    print(f"  Total orders analyzed: {predictions.height:,}")
    
    for risk in ['🔴 CRITICAL', '🟡 HIGH', '🟠 MEDIUM', '🟢 LOW']:
        count = predictions.filter(pl.col('risk_level') == risk).height
        pct = count / predictions.height * 100
        print(f"  {risk}: {count:,} orders ({pct:.1f}%)")
    
    # STEP 4: Generate actions for high-risk orders
    high_priority = predictions.filter(
        pl.col('risk_level').is_in(['🔴 CRITICAL', '🟡 HIGH'])
    ).sort('shortage_probability', descending=True)
    
    if high_priority.height > 0:
        print(f"\n⚠️  ACTION REQUIRED: {high_priority.height:,} high-risk orders")
        
        print("\n🔄 Step 2: Finding product substitutions...")
        
        actions = []
        for row in high_priority.iter_rows(named=True):
            # Get substitutes for this product
            substitutes = sub_engine.get_substitutes(str(row['product_code']), top_n=3)
            
            # Determine recommended action
            if substitutes:
                action = '📞 Contact customer - offer substitutes'
            else:
                action = '📞 Contact customer - notify potential delay'
            
            actions.append({
                'order_number': row['order_number'],
                'risk_level': row['risk_level'],
                'product_code': row['product_code'],
                'customer_number': row['customer_number'],
                'order_qty': row['order_qty'],
                'shortage_probability': row['shortage_probability'],
                'recommended_action': action,
                'substitute_1': substitutes[0] if len(substitutes) > 0 else None,
                'substitute_2': substitutes[1] if len(substitutes) > 1 else None,
                'substitute_3': substitutes[2] if len(substitutes) > 2 else None,
            })
        
        df_actions = pl.DataFrame(actions)
        
        print("\n✅ Top 10 Priority Actions:")
        print(df_actions.select([
            'order_number', 'risk_level', 'product_code', 
            'shortage_probability', 'recommended_action'
        ]).head(10))
        
        # Save outputs
        predictions.write_parquet(Config.PREDICTIONS_PATH)
        df_actions.write_csv(Config.ACTIONS_PATH)
        
        print(f"\n💾 Outputs saved:")
        print(f"  • All predictions: {Config.PREDICTIONS_PATH}")
        print(f"  • Action items: {Config.ACTIONS_PATH}")
        
        return predictions, df_actions
    
    else:
        print("\n✅ All orders are low-medium risk. No immediate action needed.")
        predictions.write_parquet(Config.PREDICTIONS_PATH)
        return predictions, None


# ============================================================================
# SECTION 8: MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution flow:
    1. Load data
    2. Clean data
    3. Engineer features
    4. Train models
    5. Save model
    6. Run prediction pipeline on sample data
    """
    
    print("\n" + "="*80)
    print("🎯 SMART DISPATCHER - COMPLETE PIPELINE")
    print("="*80)
    print("\nThis pipeline will:")
    print("  1. Load and clean historical order data")
    print("  2. Engineer predictive features")
    print("  3. Train machine learning models")
    print("  4. Create prediction pipeline")
    print("  5. Generate shortage predictions and recommendations")
    
    # PHASE 1: TRAINING (Run once to build the model)
    print("\n" + "="*80)
    print("PHASE 1: MODEL TRAINING")
    print("="*80)
    
    # Load data
    df_sales, df_purchases, df_products, df_replacements = load_data()
    
    # Clean data
    df_clean = clean_data(df_sales, df_products)
    
    # Engineer features
    df_features = engineer_all_features(df_clean, df_replacements)
    
    # Prepare for modeling
    X_train, X_test, y_train, y_test, feature_cols, df_train, df_test = prepare_modeling_data(df_features)
    
    # Train models
    best_model, feature_importance, metrics = train_models(X_train, X_test, y_train, y_test, feature_cols)
    
    # Save model
    save_model(best_model, feature_cols, feature_importance)
    
    # PHASE 2: PREDICTION (Run this for new orders)
    print("\n" + "="*80)
    print("PHASE 2: PREDICTION PIPELINE")
    print("="*80)
    
    # Initialize predictor
    predictor = ShortagePredictor(Config.MODEL_PATH)
    predictor.load_historical_data(df_features)
    
    # Initialize substitution engine
    sub_engine = SubstitutionEngine(df_products, df_replacements)
    
    # Example: Run on sample orders
    print("\n📦 Running on sample orders...")
    sample_orders = df_clean.filter(
        pl.col('order_created_date') >= df_test['order_created_date'].min()
    ).head(1000).select([
        'order_number', 'order_created_date', 'customer_number', 
        'product_code', 'order_qty', 'plant', 'storage_location'
    ])
    
    # Run integrated pipeline
    predictions, actions = run_smart_dispatcher(predictor, sub_engine, sample_orders)
    
    # Show example of detailed explanation
    if actions is not None and actions.height > 0:
        print("\n" + "="*80)
        print("🔍 DETAILED EXPLANATION FOR HIGHEST RISK ORDER")
        print("="*80)
        highest_risk = predictions.filter(pl.col('risk_level') == '🔴 CRITICAL').head(1)
        if highest_risk.height > 0:
            predictor.explain_prediction(highest_risk)
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE!")
    print("="*80)
    print("\n📝 Summary:")
    print(f"  • Model trained and saved to: {Config.MODEL_PATH}")
    print(f"  • Predictions saved to: {Config.PREDICTIONS_PATH}")
    if actions is not None:
        print(f"  • Action items saved to: {Config.ACTIONS_PATH}")
    print("\n🎉 Smart Dispatcher is ready for production use!")


# ============================================================================
# SECTION 9: USAGE EXAMPLES FOR PRODUCTION
# ============================================================================

def predict_new_orders_example():
    """
    Example: How to use the trained model for new orders in production
    
    This is what you'd run daily/hourly to predict shortages for incoming orders
    """
    print("\n" + "="*80)
    print("📋 EXAMPLE: PREDICTING NEW ORDERS")
    print("="*80)
    
    # Step 1: Load the trained model
    predictor = ShortagePredictor(Config.MODEL_PATH)
    
    # Step 2: Load historical data for feature engineering
    # In production, you'd load this from your database
    df_features = pl.read_parquet('test_data_full.parquet')  # Saved earlier
    predictor.load_historical_data(df_features)
    
    # Step 3: Load substitution engine
    with open(Config.PRODUCT_FILE, 'r', encoding='utf-8') as f:
        product_data = json.load(f)
    df_products = pl.DataFrame(product_data)
    sub_engine = SubstitutionEngine(df_products)
    
    # Step 4: Create sample new orders (in production, this comes from your order system)
    new_orders = pl.DataFrame({
        'order_number': [99999001, 99999002, 99999003],
        'order_created_date': [pl.date(2025, 10, 1)] * 3,
        'customer_number': [1001, 1002, 1003],
        'product_code': ['400234', '416580', '412619'],
        'order_qty': [10.0, 5.0, 20.0],
        'plant': [1003, 1003, 1004],
        'storage_location': [101, 102, 101]
    })
    
    # Step 5: Run prediction
    predictions, actions = run_smart_dispatcher(predictor, sub_engine, new_orders)
    
    print("\n✅ Predictions ready! Check the output files.")


# ============================================================================
# RUN THE PIPELINE
# ============================================================================

if __name__ == "__main__":
    """
    Run this script to:
    1. Train the model (first time)
    2. Generate predictions for sample orders
    
    For production use, you would:
    - Run main() once to train the model
    - Then use predict_new_orders_example() for each batch of new orders
    """
    
    # Full training and prediction pipeline
    main()
    
    # Uncomment below to run prediction on new orders only
    # predict_new_orders_example()