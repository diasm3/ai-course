import streamlit as st
import io
import os
from PIL import Image
from openai import OpenAI

# Set title
st.title("Multi-Image ChatBot")

# Initialize session state for storing chat history and images
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "images" not in st.session_state:
    st.session_state.images = []

# File uploader for multiple images
uploaded_files = st.file_uploader(
    "Upload images", 
    accept_multiple_files=True,
    type=["png", "jpg", "jpeg"]
)

# Process uploaded files
if uploaded_files:
    st.session_state.images = []
    for uploaded_file in uploaded_files:
        # Read the file and convert to PIL Image
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Store the image in session state
        st.session_state.images.append({"image": image, "name": uploaded_file.name})
    
    # Display images
    if st.session_state.images:
        st.write(f"Uploaded {len(st.session_state.images)} images:")
        cols = st.columns(min(3, len(st.session_state.images)))
        for i, img_data in enumerate(st.session_state.images):
            with cols[i % 3]:
                st.image(img_data["image"], caption=img_data["name"], use_column_width=True)

# User input
user_input = st.text_input("Ask a question about the uploaded images:")

# Function to get response from OpenAI API
def get_gpt_response(prompt, images):
    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Prepare the message content with images
    content = [
        {"type": "text", "text": prompt}
    ]
    
    # Add images to content
    for img_data in images:
        image = img_data["image"]
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Encode image to base64
        import base64
        base64_image = base64.b64encode(img_byte_arr).decode('utf-8')
        
        # Add image to content
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            }
        )
    
    # Make API call
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use vision model
            messages=[{"role": "user", "content": content}],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Process user input
if user_input and st.session_state.images:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Get response from GPT
    response = get_gpt_response(user_input, st.session_state.images)
    
    # Add assistant response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# Display chat history
st.write("Chat History:")
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Assistant:** {message['content']}")
    st.markdown("---")

# Add a note about OpenAI API key
st.sidebar.markdown("### Configuration")
st.sidebar.info(
    "This app requires an OpenAI API key with access to GPT-4 Vision.\n\n"
    "Set your API key as an environment variable named 'OPENAI_API_KEY' before running the app."
)
