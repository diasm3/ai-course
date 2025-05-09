import io
import base64
from PIL import Image
import streamlit as st

def resize_image(image, max_dimension=1024):
    """
    Resize an image if any dimension exceeds the specified maximum dimension.
    
    Args:
        image: PIL Image object
        max_dimension: Maximum width or height
        
    Returns:
        Resized PIL Image object
    """
    if max(image.size) > max_dimension:
        ratio = max_dimension / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        return image.resize(new_size, Image.LANCZOS)
    return image

def image_to_bytes(image, format=None):
    """
    Convert a PIL Image to bytes.
    
    Args:
        image: PIL Image object
        format: Image format (if None, uses the original format)
        
    Returns:
        Bytes representation of the image
    """
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=format if format else (image.format or 'JPEG'))
    return img_byte_arr.getvalue()

def encode_image_to_base64(image_bytes):
    """
    Encode image bytes to base64 string.
    
    Args:
        image_bytes: Bytes representation of the image
        
    Returns:
        Base64 encoded string
    """
    return base64.b64encode(image_bytes).decode('utf-8')

def get_image_format(file_name):
    """
    Get the image format from a file name.
    
    Args:
        file_name: File name with extension
        
    Returns:
        Image format string
    """
    extension = file_name.split('.')[-1].lower()
    if extension in ('jpg', 'jpeg'):
        return 'JPEG'
    elif extension == 'png':
        return 'PNG'
    elif extension == 'webp':
        return 'WEBP'
    elif extension == 'gif':
        return 'GIF'
    return None

def display_chat_message(message, avatar_url=None):
    """
    Display a styled chat message in the Streamlit app.
    
    Args:
        message: Dict containing 'role' and 'content'
        avatar_url: Optional URL for avatar image
    """
    role = message['role']
    content = message['content']
    
    is_user = role == 'user'
    
    # Create message container with styling
    container = st.container()
    with container:
        columns = st.columns([1, 10])
        
        # Avatar column
        with columns[0]:
            if is_user:
                st.markdown("👤")
            else:
                st.markdown("🤖")
        
        # Content column
        with columns[1]:
            st.markdown(
                f"<div style='background-color:{'#E0F7FA' if not is_user else '#F3F4F6'}; "
                f"padding:10px; border-radius:5px; margin-bottom:10px'>"
                f"<strong>{'You' if is_user else 'Assistant'}</strong><br>{content}"
                f"</div>",
                unsafe_allow_html=True
            )

def format_file_size(size_bytes):
    """
    Format file size in bytes to human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024 or unit == 'GB':
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
        
def create_image_gallery(images, columns=3):
    """
    Create a responsive image gallery in Streamlit.
    
    Args:
        images: List of image data dictionaries
        columns: Number of columns in the gallery
    """
    cols = st.columns(columns)
    
    for i, img_data in enumerate(images):
        with cols[i % columns]:
            st.image(img_data["image"], caption=img_data["name"], use_column_width=True)
            
            # Display additional info if available
            if "size" in img_data and "dimensions" in img_data:
                st.caption(f"Size: {img_data['size']} | {img_data['dimensions']}")