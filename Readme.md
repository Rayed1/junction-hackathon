# Valio Aimo - AI-Powered Smart Dispatcher

This project is a Streamlit web application designed for the **Valio Aimo "Zero-Fail Logistics" challenge**. It provides a powerful, interactive dashboard for predicting product shortages in real-time, explaining the predictions, and generating AI-powered customer communications. The system leverages a machine learning model, generative AI for text and voice, and a multimodal vision AI for post-delivery issue resolution.

![Demo Screenshot](https://i.imgur.com/your-screenshot-url.png)
*(**Note:** Replace the placeholder URL above with a real screenshot of your running application!)*

## 🌟 Key Features

*   **📈 Predictive Shortage Model:** Utilizes a trained **XGBoost** model to predict the probability of a product shortage for any given order line with **98.9% recall** on test data.
*   **⚡ Live Single-Order Prediction:** An interactive form to instantly check the shortage risk for any custom order by selecting a customer, product, and quantity.
*   **📊 Batch Dispatch Simulation:** A dashboard that processes a batch of 1,000 new orders, categorizing them by risk level (Critical, High, Medium, Low) and providing a prioritized action list.
*   **🤖 Generative AI Communication:**
    *   For high-risk orders, the app uses **OpenRouter** to generate a professional, friendly phone call script to inform the customer.
    *   It then uses the **ElevenLabs API** to convert this script into a realistic voice call, which can be played directly in the app.
*   **📸 Multimodal AI Inspector:**
    *   Allows a user to simulate a customer reporting a delivery issue by uploading an image.
    *   A **Vision AI model** (Google Gemini Pro Vision via OpenRouter) analyzes the image to identify the product.
    *   The system then immediately checks the *current* predicted shortage risk for that identified product, closing the logistics feedback loop.
*   **🧠 Explainable AI (XAI):** For any high-risk prediction, the app provides a "Deep Dive" section that explains the **Top 10 Risk Factors** that contributed to the model's decision.
*   **⚙️ Developer Mode:** A sidebar with debugging tools, including an API mocking toggle (to prevent costs during development) and a session state inspector.

## 🛠️ Tech Stack

*   **Data Science & ML:**
    *   [Polars](https://pola.rs/): For high-performance data manipulation.
    *   [Pandas](https://pandas.pydata.org/): For data structuring and compatibility.
    *   [Scikit-learn](https://scikit-learn.org/): For model evaluation metrics.
    *   [XGBoost](https://xgboost.ai/): For the gradient boosting prediction model.

*   **Generative AI Services:**
    *   [OpenRouter](https://openrouter.ai/): Acts as a gateway to various large language models (LLMs) and vision models.
        *   **Text Generation:** `nousresearch/nous-hermes-2-mixtral-8x7b-dpo`
        *   **Vision Analysis:** `google/gemini-pro-vision`
    *   [ElevenLabs](https://elevenlabs.io/): For state-of-the-art, realistic text-to-speech synthesis.

*   **Web Application & Tooling:**
    *   [Streamlit](https://streamlit.io/): For building the interactive web application.
    *   [Python 3.10+](https://www.python.org/)

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### 1. Prerequisites

*   Python 3.10 or higher
*   Git (for cloning the repository)

### 2. Installation & Setup

**Step 1: Clone the repository**
```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

**Step 2: Create a virtual environment** (recommended)
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**Step 3: Install dependencies**
```bash
pip install -r requirements.txt```

**Step 4: Set up API Keys (CRITICAL)**
This project requires API keys for OpenRouter and ElevenLabs. We use Streamlit's built-in secrets management.

1.  Create a folder named `.streamlit` in the root of your project directory.
2.  Inside `.streamlit`, create a file named `secrets.toml`.
3.  Add your keys to this file in the following format:

    ```toml
    # .streamlit/secrets.toml

    OPENROUTER_API_KEY = "sk-or-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    ELEVENLABS_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    ```
    > **⚠️ IMPORTANT:** Never commit the `secrets.toml` file to a public Git repository. The `.gitignore` file should include `.streamlit/secrets.toml`.

**Step 5: Create the audio directory**
The application will save generated audio files to a local directory. Create this folder in the project root:
```bash
mkdir generated_audio
```

### 3. Running the Application

Once the setup is complete, run the following command in your terminal:
```bash
streamlit run demo_app.py
```
Your web browser will automatically open with the application running.

## 📖 How to Use the Application

1.  **Live Single-Order Predictor:** Use the form at the top to select a customer, product, and other details. Click "Predict Shortage Risk" to see an instant analysis.
2.  **Daily Dispatch Simulation:** Click the large "Run Smart Dispatcher" button. This will process a sample batch of 1,000 orders and populate the dashboard below.
3.  **Generate AI Calls:** In the "Priority Action Items" table, check the box in the "Generate AI Call" column for any high-risk order. A spinner will appear, and once complete, an expander with the generated script and audio player will show up in the "Generated AI Customer Calls" section.
4.  **Explain a Prediction:** In the "Order Deep Dive" section, use the dropdown to select any high-risk order. The dashboard will update to show you the top risk factors and visual suggestions for substitutes.
5.  **Test the Vision AI:** Scroll down to the "Post-Delivery AI Inspector". Upload an image of a product (e.g., milk, cheese). Click "Analyze Image" to see the AI identify the product and check its current shortage risk.
6.  **Use Debug Mode:** Open the sidebar on the left to access developer tools. You can enable "Mock API Calls" to test the UI without cost, or enable "Debug Mode" to see detailed logs and AI prompts.

## 📁 Project Structure
```
.
├── .streamlit/
│   └── secrets.toml        # <-- For API keys (CRITICAL, DO NOT COMMIT)
├── data/
│   └── ... (CSV/JSON data files for the hackathon)
├── generated_audio/        # <-- AI-generated voice calls are saved here
├── shortage_predictor_model.pkl # <-- The trained XGBoost model
├── demo_app.py             # <-- The main Streamlit application script
├── requirements.txt        # <-- Project dependencies
└── README.md               # <-- You are here
```