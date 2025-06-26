# 🚀 Quick Setup Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Get API Keys

### For Moondream (Required for Moondream mode):
1. Go to [console.moondream.ai](https://console.moondream.ai)
2. Create an account and get your API key
3. Copy the API key

### For Gemini (Required for Gemini mode):
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the API key

## Step 3: Configure Secrets

1. **Copy the example secrets file:**
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```

2. **Edit `.streamlit/secrets.toml` and add your keys:**
   ```toml
   # Add your actual API keys here
   GOOGLE_API_KEY = "your-actual-google-api-key"
   MOONDREAM_API_KEY = "your-actual-moondream-api-key"
   ```

## Step 4: Run the App

```bash
streamlit run app.py
```

## 🎉 That's it!

- The app will open at `http://localhost:8501`
- Choose between Moondream API or Gemini API in the sidebar
- Upload images/videos or use your webcam
- Start analyzing!

## 💡 Pro Tips

- **Both models are now cloud APIs** - no local server setup needed
- **Moondream** is great for detailed image analysis and object detection
- **Gemini** is powerful for general-purpose vision tasks
- You can switch between models anytime in the sidebar 