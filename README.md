# 👁️ Multimodal VLM Analysis

A powerful Streamlit application for analyzing images, videos, and live webcam feeds using state-of-the-art multimodal Visual Language Models (VLMs). Supports both local inference with Moondream and cloud-based analysis with Google Gemini.

## ✨ Features

- **🖼️ Image Analysis**: Upload and analyze images with custom prompts
- **🎥 Video Analysis**: Process video files frame by frame with configurable intervals
- **📷 Live Webcam**: Real-time analysis of webcam feeds
- **🤖 Dual Model Support**: 
  - **Moondream (Local)**: Run inference locally for privacy and offline use
  - **Gemini (API)**: Use Google's powerful cloud-based model
- **📝 Multiple Task Types**: General captioning, specific questions, and OCR transcription
- **⚙️ Configurable Settings**: Adjustable analysis intervals and model parameters

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- For local Moondream: CUDA-capable GPU (optional, but recommended)
- For Gemini API: Google API key

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd moondream
   ```

2. **Install dependencies with uv (recommended):**
   ```bash
   uv sync
   ```

   Or with pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys (for Gemini):**
   
   Create a `.streamlit/secrets.toml` file:
   ```toml
   GOOGLE_API_KEY = "your-google-api-key-here"
   ```

   To get a Google API key:
   1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   2. Create a new API key
   3. Add it to your secrets file

### Running the App

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## 📖 Usage Guide

### 1. Choose Your Model

**Moondream (Local):**
- ✅ Privacy-focused (runs locally)
- ✅ No internet required after setup
- ✅ Free to use
- ❌ Requires more computational resources
- ❌ Slower inference on CPU

**Gemini (API):**
- ✅ Fast inference
- ✅ High-quality responses
- ✅ No local computational requirements
- ❌ Requires internet connection
- ❌ API costs apply
- ❌ Data sent to Google servers

### 2. Select Input Source

**File Upload:**
- Supports images (PNG, JPG, JPEG) and videos (MP4, MOV, AVI)
- For videos, analysis runs at configurable intervals
- Results are displayed with frame-by-frame review

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
├── moondream_interface.py # Moondream model wrapper
├── gemini_interface.py    # Gemini API wrapper
├── pyproject.toml        # Project dependencies (uv)
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

### Adding New Models

To add support for additional VLMs:

1. Create a new interface file (e.g., `new_model_interface.py`)
2. Implement the `answer_question(image, question)` method
3. Add the model option to the sidebar in `app.py`
4. Update the model loading logic

### Environment Variables

- `GOOGLE_API_KEY`: Your Google Gemini API key (for Gemini mode)

## 🔧 Troubleshooting

### Common Issues

**Moondream Model Loading Fails:**
- Ensure you have sufficient RAM (8GB+ recommended)
- For GPU acceleration, install CUDA and PyTorch with CUDA support
- Try running on CPU if GPU memory is insufficient

**Gemini API Errors:**
- Verify your API key is correct and has sufficient quota
- Check your internet connection
- Ensure the API key is properly set in `.streamlit/secrets.toml`

**Webcam Not Working:**
- Allow camera permissions in your browser
- Try refreshing the page
- Check if another application is using the camera

**Video Analysis Issues:**
- Ensure video format is supported (MP4, MOV, AVI)
- Large videos may take time to process
- Consider reducing analysis interval for faster processing

### Performance Tips

- **For Moondream**: Use GPU acceleration when available
- **For large videos**: Increase analysis interval to reduce processing time
- **For webcam**: Lower analysis frequency for smoother performance

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- [Moondream](https://github.com/vikhyatk/moondream) by Vikhyatk
- [Google Gemini](https://ai.google.dev/) by Google
- [Streamlit](https://streamlit.io/) for the web framework
- [Streamlit WebRTC](https://github.com/whitphx/streamlit-webrtc) for webcam support 