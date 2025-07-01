# 👁️ Multimodal VLM Analysis

A powerful Streamlit application for analyzing images, videos, and live webcam feeds using state-of-the-art multimodal Visual Language Models (VLMs). Supports both Moondream Cloud API and Google Gemini API for fast, reliable inference.

## ✨ Features

- **🖼️ Image Analysis**: Upload and analyze images with custom prompts
- **🎥 Video Analysis**: Process video files frame by frame with configurable intervals
- **📷 Live Webcam**: Real-time analysis of webcam feeds
- **🤖 Dual Cloud API Support**: 
  - **Moondream Cloud**: Advanced vision-language model via API
  - **Gemini API**: Google's powerful multimodal model
- **📝 Multiple Task Types**: General captioning, specific questions, and OCR transcription
- **⚙️ Configurable Settings**: Adjustable analysis intervals and model parameters

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Internet connection (for API calls)
- API keys for your chosen model(s)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd moondream
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys:**
   
   Copy the example secrets file:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```
   
   Edit `.streamlit/secrets.toml` and add your API keys:
   ```toml
   # Get from https://console.moondream.ai
   MOONDREAM_API_KEY = "your-moondream-api-key-here"
   
   # Get from https://makersuite.google.com/app/apikey
   GOOGLE_API_KEY = "your-google-api-key-here"
   ```

### Getting API Keys

**Moondream API:**
1. Go to [console.moondream.ai](https://console.moondream.ai)
2. Create an account and get your API key

**Google Gemini API:**
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key

### Running the App

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## 📖 Usage Guide

### 1. Choose Your Model

**Moondream Cloud API:**
- ✅ Specialized vision-language model
- ✅ Fast inference via cloud API
- ✅ Excellent for detailed image analysis
- ✅ Supports captioning, Q&A, detection, and pointing

**Gemini API:**
- ✅ Google's advanced multimodal model
- ✅ High-quality responses
- ✅ Strong general-purpose capabilities
- ✅ Excellent for complex reasoning tasks

### 2. Select Input Source

**File Upload:**
- Supports images (PNG, JPG, JPEG) and videos (MP4, MOV, AVI)
- For videos, analysis runs at configurable intervals
- Results displayed with frame-by-frame review

**Webcam:**
- Real-time analysis of live webcam feed
- Configurable analysis frequency
- Live results display

### 3. Configure Analysis

- **Analysis Interval**: Set how often to analyze frames (1-30 seconds)
- **Prompt Type**: Choose from predefined tasks or ask custom questions
- **Custom Prompts**: Ask specific questions about your content

## 🛠️ Development

### Project Structure

```
moondream/
├── app.py                 # Main Streamlit application
├── moondream_interface.py # Moondream Cloud API wrapper
├── gemini_interface.py    # Gemini API wrapper
├── requirements.txt       # Dependencies
├── pyproject.toml        # Modern dependency management
├── setup_guide.md        # Quick setup instructions
├── .gitignore           # Git ignore rules
├── .streamlit/
│   └── secrets.toml.example  # API key template
└── README.md            # This file
```

### Adding New Models

To add support for additional VLMs:

1. Create a new interface file (e.g., `new_model_interface.py`)
2. Implement the `answer_question(image, question)` method
3. Add the model option to the sidebar in `app.py`
4. Update the model loading logic

### Environment Variables

- `MOONDREAM_API_KEY`: Your Moondream Cloud API key
- `GOOGLE_API_KEY`: Your Google Gemini API key

## 🔧 Troubleshooting

### Common Issues

**API Key Errors:**
- Verify your API keys are correct and have sufficient quota
- Check your internet connection
- Ensure API keys are properly set in `.streamlit/secrets.toml`

**Webcam Not Working:**
- Allow camera permissions in your browser
- Try refreshing the page
- Check if another application is using the camera

**Video Analysis Issues:**
- Ensure video format is supported (MP4, MOV, AVI)
- Large videos may take time to process
- Consider reducing analysis interval for faster processing

### Performance Tips

- **For large videos**: Increase analysis interval to reduce processing time
- **For webcam**: Lower analysis frequency for smoother performance
- **API calls**: Both models are optimized for cloud inference - no local GPU needed!

## 🌟 Why Cloud APIs?

This app uses cloud APIs instead of local inference for several benefits:

- **🚀 Faster Setup**: No complex model downloads or GPU configuration
- **⚡ Better Performance**: Optimized cloud infrastructure
- **🔄 Always Updated**: Latest model versions automatically
- **💻 Lower Requirements**: Works on any device with internet
- **🛡️ Reliability**: Enterprise-grade uptime and scaling

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- [Moondream](https://moondream.ai/) for their excellent vision-language model
- [Google Gemini](https://ai.google.dev/) for their powerful multimodal AI
- [Streamlit](https://streamlit.io/) for the amazing web framework
- [Streamlit WebRTC](https://github.com/whitphx/streamlit-webrtc) for webcam support 