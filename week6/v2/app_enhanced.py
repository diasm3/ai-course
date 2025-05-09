import streamlit as st
import io
import os
from PIL import Image
from dotenv import load_dotenv
import time

# Import custom modules
from utils import resize_image, image_to_bytes, display_chat_message, create_image_gallery
from chat_functions import get_openai_client, get_gpt_vision_response

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Multi-Image Vision Chat",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton button {
        width: 100%;
    }
    .gallery-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        justify-content: center;
    }
    .st-emotion-cache-1v0mbdj.e115fcil1 {
        width: 100%;
    }
    .chat-header {
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        font-size: 2.5rem;
    }
    .token-counter {
        font-size: 0.8rem;
        color: #888;
        text-align: center;
    }
    .image-info {
        text-align: center;
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }
    hr {
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    .sample-question {
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize OpenAI client
client = get_openai_client()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "images" not in st.session_state:
    st.session_state.images = []

if "image_bytes" not in st.session_state:
    st.session_state.image_bytes = []

if "show_image_details" not in st.session_state:
    st.session_state.show_image_details = False

if "token_count" not in st.session_state:
    st.session_state.token_count = 0

if "upload_time" not in st.session_state:
    st.session_state.upload_time = None

# Helper functions
def clear_chat_history():
    st.session_state.chat_history = []
    st.session_state.token_count = 0

def clear_images():
    st.session_state.images = []
    st.session_state.image_bytes = []
    st.session_state.upload_time = None

def toggle_image_details():
    st.session_state.show_image_details = not st.session_state.show_image_details

def process_uploaded_files(uploaded_files):
    """Process uploaded image files"""
    if not uploaded_files:
        return
    
    # Clear existing images
    clear_images()
    
    # Store upload time
    st.session_state.upload_time = time.strftime("%Y-%m-%d %H:%M:%S")
    
    with st.spinner("Processing uploaded images..."):
        for uploaded_file in uploaded_files:
            # Read image bytes
            image_bytes = uploaded_file.getvalue()
            
            # Convert to PIL Image for display and processing
            try:
                image = Image.open(io.BytesIO(image_bytes))
                
                # Resize if too large (to save memory and API costs)
                image = resize_image(image, max_dimension=1024)
                
                # Convert back to bytes after resizing
                image_bytes = image_to_bytes(image)
                
                # Store image and bytes
                st.session_state.images.append({
                    "image": image,
                    "name": uploaded_file.name,
                    "size": f"{len(image_bytes) / 1024:.1f} KB",
                    "dimensions": f"{image.width} x {image.height}"
                })
                st.session_state.image_bytes.append(image_bytes)
            except Exception as e:
                st.error(f"Error processing image {uploaded_file.name}: {str(e)}")

def handle_user_input(user_input):
    """Process user input and get AI response"""
    if not user_input.strip():
        return
    
    # Check if images are uploaded
    if not st.session_state.image_bytes:
        st.warning("Please upload at least one image before asking questions.")
        return
    
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Get AI response
    response_data = get_gpt_vision_response(
        user_input, 
        st.session_state.image_bytes,
        client=client
    )
    
    # Add assistant response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response_data["text"]})
    
    # Update token count
    st.session_state.token_count += response_data["tokens_used"]
    
    # Clear input
    st.session_state.user_input = ""

# Main layout
def render_sidebar():
    """Render the sidebar content"""
    st.sidebar.markdown("<h1 style='text-align: center'>🖼️ Image Chat</h1>", unsafe_allow_html=True)
    
    # File uploader
    uploaded_files = st.sidebar.file_uploader(
        "Upload images to chat about",
        accept_multiple_files=True,
        type=["png", "jpg", "jpeg", "webp"],
        help="Upload one or more images to analyze"
    )
    
    # Process uploaded files
    if uploaded_files:
        process_uploaded_files(uploaded_files)
    
    # Display upload time if available
    if st.session_state.upload_time:
        st.sidebar.caption(f"Uploaded at: {st.session_state.upload_time}")
    
    # Buttons for clearing
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.button("Clear Chat", on_click=clear_chat_history, use_container_width=True)
    with col2:
        st.button("Clear Images", on_click=clear_images, use_container_width=True)
    
    # Token usage info
    if st.session_state.token_count > 0:
        st.sidebar.markdown(
            f"<div class='token-counter'>Total tokens used: {st.session_state.token_count}</div>", 
            unsafe_allow_html=True
        )
    
    # Sample questions
    st.sidebar.markdown("### Sample Questions")
    sample_questions = [
        "What do you see in these images?",
        "What are the similarities between these images?",
        "What are the differences between these images?",
        "Describe the environment in these images.",
        "What colors are most prominent in these images?",
    ]
    
    for question in sample_questions:
        if st.sidebar.button(
            question, 
            key=f"sample_{question}"
        ):
            # Set user input
            st.session_state.user_input = question
            # Process input
            handle_user_input(question)
            # Force refresh
            st.rerun()
    
    # API information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("""
    This app uses OpenAI's GPT-4 Vision model to analyze images and answer questions.
    
    Upload images, then ask questions about them. The app remembers context throughout your conversation.
    """)

def render_image_gallery():
    """Render the image gallery"""
    if not st.session_state.images:
        return
    
    # Image gallery header with toggle button
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"### 📸 Uploaded Images ({len(st.session_state.images)})")
    with col2:
        st.button(
            "Toggle Details" if not st.session_state.show_image_details else "Hide Details",
            on_click=toggle_image_details,
            key="toggle_details"
        )
    
    # Create image gallery
    create_image_gallery(st.session_state.images, columns=min(3, len(st.session_state.images)))
    
    # Divider
    st.markdown("---")

def render_chat_interface():
    """Render the chat interface"""
    st.markdown("<h1 class='chat-header'>Multi-Image Vision Chat</h1>", unsafe_allow_html=True)
    
    # Display images
    render_image_gallery()
    
    # Display initial message if no chat history and images are uploaded
    if not st.session_state.chat_history and st.session_state.images:
        st.info("👋 Images uploaded! Ask me anything about them.")
    
    # Display empty state message
    if not st.session_state.images:
        st.markdown(
            """
            <div style="text-align: center; padding: 2rem;">
                <h2>👋 Welcome to Multi-Image Vision Chat!</h2>
                <p>Upload one or more images in the sidebar to get started.</p>
                <p>You can then ask questions about the images and I'll analyze them for you.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        return
    
    # Chat container
    chat_container = st.container()
    with chat_container:
        # Display chat history
        for message in st.session_state.chat_history:
            display_chat_message(message)
    
    # User input
    user_input = st.text_input(
        "",
        key="user_input",
        placeholder="Ask something about the images...",
        label_visibility="collapsed"
    )
    
    # Process user input
    if user_input:
        handle_user_input(user_input)
        # Force refresh
        st.rerun()

# Render the UI
render_sidebar()
render_chat_interface()