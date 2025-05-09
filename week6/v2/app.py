import streamlit as st
import io
import os
import base64
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Multi-Image Chat Assistant",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem; 
        border-radius: 0.5rem; 
        margin-bottom: 1rem; 
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #ddeeff;
    }
    .chat-message.assistant {
        background-color: #f0f0f0;
    }
    .chat-message .avatar {
        width: 20%;
    }
    .chat-message .content {
        width: 80%;
        padding-left: 1rem;
    }
    .stButton button {
        width: 100%;
    }
    .thumbnail-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .thumbnail {
        flex: 0 0 auto;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Get OpenAI API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("⚠️ OpenAI API key not found! Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

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

def clear_chat_history():
    st.session_state.chat_history = []
    st.session_state.token_count = 0

def clear_images():
    st.session_state.images = []
    st.session_state.image_bytes = []

def toggle_image_details():
    st.session_state.show_image_details = not st.session_state.show_image_details

def get_gpt_response(prompt, images):
    """Get response from OpenAI API with image context"""
    # Prepare message content
    content = [
        {"type": "text", "text": prompt}
    ]
    
    # Add images to content
    for img_bytes in images:
        base64_image = base64.b64encode(img_bytes).decode('utf-8')
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })
    
    try:
        with st.spinner("🤔 Thinking..."):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that can see and analyze multiple images. Provide detailed and accurate information about what you see in the images."},
                    {"role": "user", "content": content}
                ],
                max_tokens=500
            )
        
        # Update token usage
        tokens_used = response.usage.total_tokens
        st.session_state.token_count += tokens_used
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Sidebar
with st.sidebar:
    st.title("🖼️ Multi-Image Chat")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload images to chat about",
        accept_multiple_files=True,
        type=["png", "jpg", "jpeg", "webp"],
    )
    
    # Process uploaded files
    if uploaded_files:
        clear_images()  # Clear existing images
        for uploaded_file in uploaded_files:
            # Read image bytes
            image_bytes = uploaded_file.getvalue()
            
            # Convert to PIL Image for display
            image = Image.open(io.BytesIO(image_bytes))
            
            # Resize if too large (to save memory and API costs)
            max_dim = 1024
            if max(image.size) > max_dim:
                ratio = max_dim / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.LANCZOS)
                
                # Convert back to bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
                image_bytes = img_byte_arr.getvalue()
            
            # Store image and bytes
            st.session_state.images.append({
                "image": image,
                "name": uploaded_file.name,
                "size": f"{len(image_bytes) / 1024:.1f} KB",
                "dimensions": f"{image.width} x {image.height}"
            })
            st.session_state.image_bytes.append(image_bytes)
    
    # Buttons for clearing
    col1, col2 = st.columns(2)
    with col1:
        st.button("Clear Chat", on_click=clear_chat_history, use_container_width=True)
    with col2:
        st.button("Clear Images", on_click=clear_images, use_container_width=True)
    
    # Token usage info
    if st.session_state.token_count > 0:
        st.info(f"Total tokens used: {st.session_state.token_count}")
    
    # Sample questions
    st.markdown("### Sample Questions")
    sample_questions = [
        "What do you see in these images?",
        "What are the similarities between these images?",
        "What are the differences between these images?",
        "Can you describe the environment in these images?",
        "What emotions do these images evoke?",
    ]
    
    for question in sample_questions:
        if st.button(question, key=f"sample_{question}"):
            # Add to text input
            st.session_state.user_input = question
    
    # API information
    st.markdown("### About")
    st.markdown("""
    This app uses OpenAI's GPT-4 Vision model to analyze images and respond to questions.
    
    Upload images, then ask questions about them in the chat.
    """)

# Main content
st.title("Multi-Image Chat Assistant")

# Display uploaded images if any
if st.session_state.images:
    # Image gallery header with toggle button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"📸 Uploaded Images ({len(st.session_state.images)})")
    with col2:
        st.button(
            "Toggle Details" if not st.session_state.show_image_details else "Hide Details",
            on_click=toggle_image_details,
            key="toggle_details"
        )
    
    # Image gallery
    image_cols = st.columns(min(3, len(st.session_state.images)))
    for i, img_data in enumerate(st.session_state.images):
        with image_cols[i % 3]:
            st.image(img_data["image"], caption=img_data["name"], use_column_width=True)
            if st.session_state.show_image_details:
                st.caption(f"Size: {img_data['size']} | Dimensions: {img_data['dimensions']}")
    
    st.divider()
else:
    st.info("👈 Please upload one or more images using the sidebar to begin.")

# Chat interface
if st.session_state.images:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.container():
            st.markdown(f"""
            <div class="chat-message {message['role']}">
                <div class="content">
                    <p><strong>{"You" if message['role'] == 'user' else "Assistant"}:</strong> {message['content']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # User input
    user_input = st.text_input(
        "Ask about the images:",
        key="user_input",
        placeholder="What would you like to know about these images?"
    )
    
    # Process user input
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get AI response
        if st.session_state.image_bytes:
            response = get_gpt_response(user_input, st.session_state.image_bytes)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Rerun to update UI
        st.rerun()