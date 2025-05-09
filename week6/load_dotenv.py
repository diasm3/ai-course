import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check if the API key is loaded
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("Warning: OPENAI_API_KEY not found in environment variables!")
    print("Please create a .env file based on .env.example with your API key.")
else:
    print("API key loaded successfully.")