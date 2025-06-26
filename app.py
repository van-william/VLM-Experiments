import streamlit as st
from PIL import Image
import os
import cv2
import numpy as np
import tempfile
import queue
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# --- Helper Functions to Check for Interface Files ---
def check_file_exists(path):
    if not os.path.exists(path):
        st.error(f"Error: Required file not found at '{path}'. Please create it.")
        st.stop()

# --- Check for interface files before importing ---
check_file_exists('moondream_interface.py')
check_file_exists('gemini_interface.py')

from moondream_interface import MoondreamInterface
from gemini_interface import GeminiInterface

# --- Page Configuration ---
st.set_page_config(
    page_title="Multimodal VLM Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("👁️ Multimodal Visual Language Model Analysis")
st.markdown("A tool to analyze images, videos, and live webcam feeds using either a local Moondream model or the Google Gemini API.")

# --- Model Loading (Cached) ---
@st.cache_resource
def load_moondream_model():
    """Loads the Moondream model and caches it."""
    try:
        api_key = st.secrets.get("MOONDREAM_API_KEY")
        if not api_key:
            return "API_KEY_NOT_FOUND"
        return MoondreamInterface(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to load Moondream model. Error: {e}")
        return None

@st.cache_resource
def load_gemini_client():
    """Initializes the Gemini client and caches it."""
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if not api_key:
            return "API_KEY_NOT_FOUND"
        return GeminiInterface(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize Gemini client. Error: {e}")
        return None

# --- Session State Initialization ---
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []
if 'current_file' not in st.session_state:
    st.session_state.current_file = None

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    model_choice = st.radio(
        "Choose the model to use:",
        ("Moondream (API)", "Gemini (API)"),
        help="Select 'Moondream' to use the Moondream Cloud API. Select 'Gemini' to use the Google Cloud API."
    )
    
    st.markdown("---")
    st.header("📹 Input Settings")
    input_source = st.radio("Select Input Source:", ("File Upload", "Webcam"))
    
    analysis_interval = st.slider(
        "Analysis interval (seconds):", 
        min_value=1, max_value=30, value=5,
        help="For video files or webcam, the model will analyze one frame every N seconds."
    )

# --- Model Selection Logic ---
model = None
if model_choice == "Moondream (API)":
    model = load_moondream_model()
    if model == "API_KEY_NOT_FOUND":
        st.error("Moondream API Key not found. Please add it to your Streamlit secrets.", icon="🚨")
        st.info("Create a file named `.streamlit/secrets.toml` with: `MOONDREAM_API_KEY = \"YOUR_KEY\"`")
        st.stop()
    elif model is None:
        st.error("Moondream client failed to load. Check your connection and API key.")
        st.stop()
else: # Gemini (API)
    model = load_gemini_client()
    if model == "API_KEY_NOT_FOUND":
        st.error("Google API Key not found. Please add it to your Streamlit secrets.", icon="🚨")
        st.info("Create a file named `.streamlit/secrets.toml` with: `GOOGLE_API_KEY = \"YOUR_KEY\"`")
        st.stop()
    elif model is None:
        st.error("Gemini client failed to load. Check your connection and API key.")
        st.stop()

# --- Main UI ---
if input_source == "File Upload":
    uploaded_file = st.file_uploader(
        "Upload an image or video for analysis",
        type=["png", "jpg", "jpeg", "mp4", "mov", "avi"]
    )

    if uploaded_file and uploaded_file.name != st.session_state.current_file:
        st.session_state.analysis_results = []
        st.session_state.current_file = uploaded_file.name

    if uploaded_file:
        file_type = uploaded_file.type.split('/')[0]
        if file_type == "image":
            # Image analysis logic (unchanged)
            col1, col2 = st.columns(2)
            image = Image.open(uploaded_file).convert("RGB")
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            with col2:
                st.subheader("💬 Prompt")
                prompt_type = st.selectbox("Select a task type:", ("General Caption", "Ask a Specific Question", "Transcribe Document (OCR)"), key="img_prompt_type")
                user_prompt = ""
                if prompt_type == "General Caption": user_prompt = "Describe this image in detail."
                elif prompt_type == "Transcribe Document (OCR)": user_prompt = "Transcribe the text in natural reading order."
                else: user_prompt = st.text_input("Enter your question:", key="img_user_question")
                
                if st.button("Analyze Image", use_container_width=True, type="primary"):
                    if model and user_prompt:
                        with st.spinner(f"Analyzing with {model_choice}..."):
                            answer = model.answer_question(image, user_prompt)
                            st.subheader("🧠 Model Response")
                            st.markdown(answer)
                    else:
                        st.warning("Please provide a prompt and ensure a model is loaded.")
        elif file_type == "video":
            # Video analysis logic (unchanged)
            st.subheader("📹 Video Analysis")
            prompt = st.text_input("What should I look for in the video?", "Describe what is happening in this frame.", key="video_prompt")
            
            if st.button("Analyze Video", use_container_width=True, type="primary"):
                if model and prompt:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                        tfile.write(uploaded_file.read())
                        video_path = tfile.name

                    vid_cap = cv2.VideoCapture(video_path)
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    frame_interval_v = int(fps * analysis_interval)
                    frame_count = 0; results = []
                    progress_bar = st.progress(0, text="Analyzing video frames...")
                    while vid_cap.isOpened():
                        ret, frame = vid_cap.read()
                        if not ret: break
                        if frame_count % frame_interval_v == 0:
                            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            answer = model.answer_question(pil_image, prompt)
                            results.append({"frame_number": frame_count, "timestamp": frame_count / fps, "image": pil_image, "analysis": answer})
                            total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)); progress_bar.progress(frame_count / total_frames, text=f"Analyzing frame {frame_count}/{total_frames}")
                        frame_count += 1
                    vid_cap.release(); os.remove(video_path); progress_bar.empty()
                    st.session_state.analysis_results = results
                    if not results: st.warning("No frames were analyzed.")
                else: st.warning("Please provide a prompt and ensure a model is loaded.")

            if st.session_state.analysis_results:
                st.markdown("---"); st.subheader("🔍 Review Analysis")
                result_count = len(st.session_state.analysis_results); selected_frame_index = st.slider(f"Scrub through {result_count} analyzed frames:", 0, result_count - 1, 0)
                current_result = st.session_state.analysis_results[selected_frame_index]
                col1, col2 = st.columns(2)
                with col1: st.image(current_result["image"], caption=f"Frame at {current_result['timestamp']:.2f}s", use_column_width=True)
                with col2: st.markdown(f"**Analysis for frame at {current_result['timestamp']:.2f} seconds:**"); st.markdown(current_result["analysis"])

elif input_source == "Webcam":
    st.subheader("📷 Live Webcam Analysis")
    prompt = st.text_input("What should I look for in the webcam feed?", "Describe what you see.", key="webcam_prompt")
    
    result_queue = queue.Queue()

    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        
        # This is a simple way to control analysis frequency
        # For more accuracy, a time-based approach would be better
        if not hasattr(video_frame_callback, "counter"):
            video_frame_callback.counter = 0
        
        # Assuming around 30fps, this approximates the interval in seconds
        if video_frame_callback.counter % (30 * analysis_interval) == 0:
             if model and prompt:
                pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                answer = model.answer_question(pil_image, prompt)
                result_queue.put(answer)
        
        video_frame_callback.counter += 1
        return frame

    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    
    webrtc_ctx = webrtc_streamer(
        key="webcam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    if webrtc_ctx.state.playing and model and prompt:
        st.markdown("---")
        st.subheader("🧠 Live Analysis")
        
        text_placeholder = st.empty()
        while webrtc_ctx.state.playing:
            try:
                result = result_queue.get(timeout=1.0)
                text_placeholder.info(result)
            except queue.Empty:
                pass
    elif not model:
        st.warning("Model is not loaded. Please select one from the sidebar.")
    elif not prompt:
        st.info("Enter a prompt above and the analysis will appear here.")
