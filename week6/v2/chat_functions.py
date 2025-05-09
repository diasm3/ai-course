import streamlit as st
from openai import OpenAI
import os
from utils import encode_image_to_base64

def get_openai_client():
    """
    Initialize and ryour_openai_api_key_hereeturn the OpenAI client using the API key from environment variables.
    
    Returns:
        OpenAI client instance
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ OpenAI API key not found! Please set the OPENAI_API_KEY environment variable.")
        st.stop()
    
    return OpenAI(api_key=api_key)

def create_message_with_images(prompt, image_bytes_list):
    """
    Create a message with text and images for the OpenAI API.
    
    Args:
        prompt: Text prompt for the model
        image_bytes_list: List of image bytes to include
        
    Returns:
        List of content objects for the OpenAI API
    """
    content = [{"type": "text", "text": prompt}]
    
    for img_bytes in image_bytes_list:
        base64_image = encode_image_to_base64(img_bytes)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })
    
    return content

def get_system_prompt():
    """
    Get the system prompt for the model.
    
    Returns:
        System prompt string
    """
    return """You are a helpful and detailed image analysis assistant.
    
When analyzing multiple images:
1. Look carefully at each image's content and details
2. When asked about similarities or differences, be specific and thorough
3. When describing images, mention colors, objects, subjects, environment, and mood
4. If there are people in the images, describe them respectfully and avoid assumptions
5. When uncertain about something in an image, acknowledge your uncertainty
6. Respond in a conversational, helpful tone

Your goal is to provide insightful analysis that helps the user understand what's in their images."""

def get_gpt_vision_response(prompt, image_bytes_list, client=None, model="gpt-4o-mini", max_tokens=500):
    """
    Get a response from GPT-4 Vision API with the given prompt and images.
    
    Args:
        prompt: Text prompt for the model
        image_bytes_list: List of image bytes
        client: OpenAI client (if None, a new one will be created)
        model: Model to use
        max_tokens: Maximum tokens for the response
        
    Returns:
        Response text and token usage
    """
    if client is None:
        client = get_openai_client()
    
    content = create_message_with_images(prompt, image_bytes_list)
    
    try:
        with st.spinner("🤔 Analyzing images..."):
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": get_system_prompt()},
                    {"role": "user", "content": content}
                ],
                max_tokens=max_tokens
            )
        
        return {
            "text": response.choices[0].message.content,
            "tokens_used": response.usage.total_tokens
        }
    except Exception as e:
        st.error(f"Error calling OpenAI API: {str(e)}")
        return {
            "text": f"I'm sorry, but I encountered an error while analyzing the images: {str(e)}",
            "tokens_used": 0
        }