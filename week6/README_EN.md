# Multi-Image ChatBot

This project implements a Streamlit-based chatbot that can process multiple images and answer questions about them using OpenAI's GPT-4 Vision API.

## Features

- Upload multiple images at once
- Ask questions about the uploaded images
- Maintain context throughout the conversation
- View chat history with all previous questions and answers

## Requirements

- Python 3.8+
- OpenAI API key with access to GPT-4 Vision
- Required Python packages (see `requirements.txt`)

## Setup

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file based on `.env.example` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Running the Application

Run the Streamlit app with:

```
streamlit run app.py
```

## Usage

1. Upload one or more images using the file uploader
2. Type your question about the images in the text input field
3. View the AI's response and continue the conversation
4. The chatbot will remember all previously uploaded images for the session

## Assignment Tasks

- [x] Accept multiple images as input
- [x] Implement free-form Q&A functionality
- [ ] Record a demonstration video showing:
  - Using dog and cat images
  - Asking about similarities between the images
  - Asking about differences between the images