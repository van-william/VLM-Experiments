import tracemalloc
tracemalloc.start()

import streamlit as st
from PIL import Image
import os
import cv2
import numpy as np
import tempfile
import queue
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
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
st.markdown("A tool to analyze images, videos, and live webcam feeds using either Moondream Cloud API or Google Gemini API.")

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
if 'focus_data' not in st.session_state:
    st.session_state.focus_data = []
if 'focus_session_start' not in st.session_state:
    st.session_state.focus_session_start = None
if 'last_focus_check' not in st.session_state:
    st.session_state.last_focus_check = 0
if 'last_ui_refresh' not in st.session_state:
    st.session_state.last_ui_refresh = 0
if 'focus_analysis_trigger' not in st.session_state:
    st.session_state.focus_analysis_trigger = 0

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # App Mode Selection
    app_mode = st.radio(
        "Choose App Mode:",
        ("General Analysis", "Focus Detection"),
        help="Select 'General Analysis' for image/video analysis or 'Focus Detection' for productivity tracking."
    )
    
    if app_mode == "General Analysis":
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
    
    elif app_mode == "Focus Detection":
        st.header("🎯 Focus Detection Settings")
        focus_model_choice = st.radio(
            "Choose focus detection model:",
            ("Moondream (API)", "Gemini (API)"),
            help="Select the AI model for focus analysis."
        )
        
        focus_interval = st.slider(
            "Focus check interval (seconds):", 
            min_value=5, max_value=30, value=10,
            help="How often to analyze focus level."
        )
        
        focus_threshold = st.slider(
            "Focus alert threshold:", 
            min_value=0, max_value=100, value=70,
            help="Alert when focus drops below this level."
        )
        
        if st.button("🗑️ Clear Focus Data"):
            st.session_state.focus_data = []
            st.session_state.focus_session_start = None
            st.rerun()

# --- Model Selection Logic ---
model = None
if app_mode == "General Analysis":
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

elif app_mode == "Focus Detection":
    if focus_model_choice == "Moondream (API)":
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

# --- Focus Detection Functions ---
def extract_focus_score(response_text):
    """Extract numeric focus score from model response."""
    import re
    
    # Convert to string if not already
    response_text = str(response_text).strip()
    
    # Try multiple patterns to extract the score
    patterns = [
        r'(?:^|\s)(\d{1,3})(?:\s|$|/100|%)',  # Number at start/end or followed by /100 or %
        r'score[:\s]*(\d{1,3})',               # "score: 85" or "score 85"
        r'focus[:\s]*(\d{1,3})',               # "focus: 85" or "focus 85"
        r'(\d{1,3})/100',                      # "85/100"
        r'(\d{1,3})%',                         # "85%"
        r'\b(\d{1,3})\b'                       # Any standalone number
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        for match in matches:
            try:
                score = int(match)
                if 0 <= score <= 100:
                    return score
            except ValueError:
                continue
    
    # If no pattern matches, try to find any number in the response
    all_numbers = re.findall(r'\d+', response_text)
    for num_str in all_numbers:
        try:
            score = int(num_str)
            if 0 <= score <= 100:
                return score
        except ValueError:
            continue
    
    return None

def get_focus_prompt():
    """Generate the focus detection prompt."""
    return """Rate this person's focus level from 0 to 100 based on:

- Eye direction and gaze
- Head position and posture
- Facial expression
- Body language and alertness

Guidelines:
- 0-30: Distracted, looking away, unfocused
- 30-60: Moderate attention, some focus
- 60-80: Good focus, engaged
- 80-100: Extremely focused and attentive

Note: Looking slightly down at a laptop screen is normal working posture.

Respond with ONLY a number from 0 to 100. No explanation needed."""

def add_focus_data_point(score, timestamp=None):
    """Add a focus score to session data."""
    if timestamp is None:
        timestamp = datetime.now()
    
    if st.session_state.focus_session_start is None:
        st.session_state.focus_session_start = timestamp
    
    st.session_state.focus_data.append({
        'timestamp': timestamp,
        'score': score,
        'session_time': (timestamp - st.session_state.focus_session_start).total_seconds()
    })

def aggregate_focus_data(interval_minutes=1):
    """Aggregate focus data by time intervals."""
    if not st.session_state.focus_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(st.session_state.focus_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Resample by interval and calculate mean
    aggregated = df.resample(f'{interval_minutes}min')['score'].mean().reset_index()
    return aggregated

# --- Main UI ---
if app_mode == "General Analysis":
    # Original analysis functionality
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

elif app_mode == "Focus Detection":
    # Focus Detection Mode
    st.header("🎯 Focus Detection Dashboard")
    
    # Initialize persistent queue in session state
    if 'focus_result_queue' not in st.session_state:
        st.session_state.focus_result_queue = queue.Queue()
        print(f"[DEBUG] ===== NEW QUEUE CREATED =====")
        print(f"[DEBUG] Queue object: {st.session_state.focus_result_queue}")
    
    focus_result_queue = st.session_state.focus_result_queue
    print(f"[DEBUG] Using queue: {focus_result_queue}")
    print(f"[DEBUG] Current queue size: {focus_result_queue.qsize()}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📷 Live Focus Monitoring")
        
        # Use a class to avoid session state issues in callback
        class FocusProcessor:
            def __init__(self, result_queue):
                self.last_analysis_time = 0
                self.frame_count = 0
                self.result_queue = result_queue
                
            def process_frame(self, frame):
                img = frame.to_ndarray(format="bgr24")
                current_time = time.time()
                self.frame_count += 1
                
                # Debug info every 30 frames (~1 second)
                if self.frame_count % 30 == 0:
                    print(f"[DEBUG] Frame #{self.frame_count}, Model: {model is not None}")
                    print(f"[DEBUG] Time since last analysis: {current_time - self.last_analysis_time:.1f}s")
                
                # Check if enough time has passed since last analysis
                # TEMPORARY: Force analysis every 30 frames for testing
                should_analyze = (current_time - self.last_analysis_time >= focus_interval) or (self.frame_count % 30 == 0 and self.frame_count > 0)
                
                if should_analyze:
                    print(f"\n[DEBUG] ====== FOCUS ANALYSIS TRIGGERED ======")
                    print(f"[DEBUG] Time: {datetime.now().strftime('%H:%M:%S')}")
                    
                    if model:
                        try:
                            print(f"[DEBUG] Converting frame to PIL image...")
                            pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                            print(f"[DEBUG] Image size: {pil_image.size}")
                            
                            focus_prompt = get_focus_prompt()
                            print(f"[DEBUG] Using prompt: {focus_prompt[:50]}...")
                            
                            print(f"[DEBUG] Sending to AI model...")
                            response = model.answer_question(pil_image, focus_prompt)
                            print(f"[DEBUG] *** RAW AI RESPONSE ***")
                            print(f"[DEBUG] Type: {type(response)}")
                            print(f"[DEBUG] Content: '{response}'")
                            print(f"[DEBUG] *** END RAW RESPONSE ***")
                            
                            focus_score = extract_focus_score(response)
                            print(f"[DEBUG] Extracted focus score: {focus_score}")
                            
                            if focus_score is not None:
                                # Put result in queue for main thread to process
                                result_data = {
                                    'score': focus_score,
                                    'timestamp': datetime.now(),
                                    'raw_response': response
                                }
                                print(f"[DEBUG] 📦 PUTTING IN QUEUE: {result_data}")
                                print(f"[DEBUG] Queue object: {self.result_queue}")
                                print(f"[DEBUG] Queue size before put: {self.result_queue.qsize()}")
                                
                                self.result_queue.put(result_data)
                                
                                print(f"[DEBUG] Queue size after put: {self.result_queue.qsize()}")
                                print(f"[DEBUG] ✅ Successfully added to queue: score {focus_score}")
                            else:
                                print(f"[DEBUG] ❌ FAILED to extract valid score!")
                                print(f"[DEBUG] Response was: '{response}'")
                                
                                # Try manual parsing
                                import re
                                all_nums = re.findall(r'\d+', str(response))
                                print(f"[DEBUG] All numbers found: {all_nums}")
                                
                                # Try to add a fallback score for testing
                                if all_nums:
                                    fallback_score = int(all_nums[0])
                                    if 0 <= fallback_score <= 100:
                                        print(f"[DEBUG] 🔄 Using fallback score: {fallback_score}")
                                        self.result_queue.put({
                                            'score': fallback_score,
                                            'timestamp': datetime.now(),
                                            'raw_response': response
                                        })
                                
                            # Update timing
                            self.last_analysis_time = current_time
                            
                        except Exception as e:
                            print(f"[DEBUG] ❌ EXCEPTION during focus detection:")
                            print(f"[DEBUG] Error: {e}")
                            import traceback
                            print(f"[DEBUG] Full traceback:")
                            print(traceback.format_exc())
                            self.last_analysis_time = current_time
                    else:
                        print(f"[DEBUG] ❌ No model available!")
                        self.last_analysis_time = current_time
                    
                    print(f"[DEBUG] ====== END FOCUS ANALYSIS ======\n")
                
                return frame
        
        # Create processor instance with the queue
        focus_processor = FocusProcessor(focus_result_queue)
        
        def focus_frame_callback(frame):
            return focus_processor.process_frame(frame)
        
        webrtc_ctx = webrtc_streamer(
            key="focus_webcam",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            video_frame_callback=focus_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    
    with col2:
        st.subheader("📊 Current Status")
        
        # Webcam status
        if webrtc_ctx.state.playing:
            st.success("🟢 Webcam Active - Focus tracking enabled")
        else:
            st.warning("🔴 Start webcam to begin focus tracking")
        
        # Test and refresh buttons
        col_test, col_refresh = st.columns(2)
        with col_test:
            if st.button("🧪 Test", help="Add test focus score"):
                # Add a fake test score to verify the system works
                import random
                test_score = random.randint(60, 95)  # Random test score
                
                # Test both direct addition and queue addition
                print(f"[DEBUG] === TEST BUTTON CLICKED ===")
                print(f"[DEBUG] Direct add to session state...")
                add_focus_data_point(test_score)
                print(f"[DEBUG] Session state now has: {len(st.session_state.focus_data)} items")
                
                # Also test queue
                print(f"[DEBUG] Adding test item to queue...")
                print(f"[DEBUG] Using queue object: {focus_result_queue}")
                focus_result_queue.put({
                    'score': test_score + 5,  # Different score to distinguish
                    'timestamp': datetime.now(),
                    'raw_response': f'Test queue item: {test_score + 5}'
                })
                print(f"[DEBUG] Queue size after test: {focus_result_queue.qsize()}")
                
                st.success(f"✅ Test score added: {test_score}/100 (+ queue test)")
                st.rerun()
        
        with col_refresh:
            if st.button("🔄 Refresh", help="Manually refresh data"):
                st.rerun()
        
        # Live focus score display
        score_placeholder = st.empty()
        alert_placeholder = st.empty()
        
        # Debug: Show data status
        data_count = len(st.session_state.focus_data)
        queue_size = focus_result_queue.qsize()
        st.write(f"**Debug**: {data_count} focus data points collected")
        st.write(f"**Queue**: {queue_size} items pending")
        st.write(f"**Webcam**: {'🟢 Playing' if webrtc_ctx.state.playing else '🔴 Stopped'}")
        st.write(f"**Model**: {'✅ Loaded' if model else '❌ None'}")
        if data_count > 0:
            st.write(f"**Latest scores**: {[d['score'] for d in st.session_state.focus_data[-5:]]}")
        
        # Session stats
        if st.session_state.focus_data:
            latest_score = st.session_state.focus_data[-1]['score']
            avg_score = np.mean([d['score'] for d in st.session_state.focus_data])
            session_duration = (datetime.now() - st.session_state.focus_session_start).total_seconds() / 60
            
            with score_placeholder.container():
                st.metric("Current Focus", f"{latest_score}/100", delta=None)
                st.metric("Session Average", f"{avg_score:.1f}/100")
                st.metric("Session Duration", f"{session_duration:.1f} min")
                
            if latest_score < focus_threshold:
                alert_placeholder.error(f"⚠️ Focus Alert! Current: {latest_score}/100")
            else:
                alert_placeholder.success("✅ Good focus level")
        else:
            score_placeholder.info("Waiting for focus data...")
    
    # Analytics Charts
    if st.session_state.focus_data:
        st.markdown("---")
        st.subheader("📈 Focus Analytics")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.subheader("1-Minute Average")
            df_1min = aggregate_focus_data(1)
            if not df_1min.empty:
                fig_1min = px.line(df_1min, x='timestamp', y='score', 
                                 title="Focus Score (1-min average)",
                                 labels={'score': 'Focus Score', 'timestamp': 'Time'})
                fig_1min.add_hline(y=focus_threshold, line_dash="dash", 
                                 line_color="red", annotation_text="Threshold")
                fig_1min.update_layout(yaxis_range=[0, 100])
                st.plotly_chart(fig_1min, use_container_width=True)
        
        with chart_col2:
            st.subheader("10-Minute Average")
            df_10min = aggregate_focus_data(10)
            if not df_10min.empty:
                fig_10min = px.line(df_10min, x='timestamp', y='score', 
                                  title="Focus Score (10-min average)",
                                  labels={'score': 'Focus Score', 'timestamp': 'Time'})
                fig_10min.add_hline(y=focus_threshold, line_dash="dash", 
                                   line_color="red", annotation_text="Threshold")
                fig_10min.update_layout(yaxis_range=[0, 100])
                st.plotly_chart(fig_10min, use_container_width=True)
        
        # Raw data table
        with st.expander("📋 Raw Focus Data"):
            df_raw = pd.DataFrame(st.session_state.focus_data)
            df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp']).dt.strftime('%H:%M:%S')
            st.dataframe(df_raw[['timestamp', 'score']].tail(20), use_container_width=True)
    
    # Process focus results from queue and auto-refresh
    if webrtc_ctx.state.playing:
        new_data_count = 0
        print(f"[DEBUG] Main thread: Checking queue... Current data count: {len(st.session_state.focus_data)}")
        print(f"[DEBUG] Queue size: {focus_result_queue.qsize()}")
        
        # Process all available results in queue
        processed_any = False
        while True:
            try:
                focus_result = focus_result_queue.get_nowait()
                # Add the result to session state (this runs in main thread)
                add_focus_data_point(focus_result['score'], focus_result['timestamp'])
                new_data_count += 1
                processed_any = True
                print(f"[DEBUG] ✅ Main thread: Added focus score {focus_result['score']} to session")
                print(f"[DEBUG] Total focus data points: {len(st.session_state.focus_data)}")
            except queue.Empty:
                break
        
        if not processed_any:
            print(f"[DEBUG] Main thread: Queue empty, no new data")
        
        # If we got new data, refresh immediately
        if new_data_count > 0:
            print(f"[DEBUG] 🔄 Auto-refreshing UI due to {new_data_count} new data points")
            st.rerun()
    
    # Always do periodic refresh to update session stats (every 3 seconds)
    current_time = time.time()
    if current_time - st.session_state.last_ui_refresh > 3:
        st.session_state.last_ui_refresh = current_time
        print("[DEBUG] ⏰ Periodic UI refresh")
        st.rerun()
