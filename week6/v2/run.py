#!/usr/bin/env python
"""
Runner script for the multi-image chatbot application.
This script checks environment variables, handles errors, and launches the Streamlit app.
"""

import os
import sys
from dotenv import load_dotenv
import subprocess
import webbrowser
from pathlib import Path

def check_environment():
    """Check if all required environment variables are set."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Check if OpenAI API key is set
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\033[91mError: OPENAI_API_KEY environment variable is not set.\033[0m")
        print("Please create a .env file with your OpenAI API key or set it in your environment.")
        print("Example .env file content:")
        print("OPENAI_API_KEY=your_openai_api_key_here")
        return False
    
    return True

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = ["streamlit", "openai", "pillow", "python-dotenv"]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"\033[91mError: Required package '{package}' is not installed.\033[0m")
            print("Please install all required packages using:")
            print("pip install -r requirements.txt")
            return False
    
    return True

def main():
    """Main entry point for the application."""
    print("\033[94m" + "="*50 + "\033[0m")
    print("\033[94m🖼️  Multi-Image Vision Chat Application\033[0m")
    print("\033[94m" + "="*50 + "\033[0m")
    
    # Check environment
    if not check_environment():
        return 1
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Determine which app file to run
    app_file = "app_enhanced.py"
    if len(sys.argv) > 1 and sys.argv[1] == "--simple":
        app_file = "app.py"
        print("\033[93mRunning in simple mode...\033[0m")
    else:
        print("\033[93mRunning in enhanced mode...\033[0m")
    
    # Get the current directory
    current_dir = Path(__file__).parent.absolute()
    app_path = current_dir / app_file
    
    # Check if app file exists
    if not app_path.exists():
        print(f"\033[91mError: App file '{app_file}' not found in {current_dir}\033[0m")
        return 1
    
    # Run the app
    print("\033[92mStarting Streamlit server...\033[0m")
    cmd = ["streamlit", "run", str(app_path)]
    
    try:
        # Open browser after a delay
        print("\033[92mOpening browser...\033[0m")
        
        # Run the process
        process = subprocess.Popen(cmd)
        process.wait()
        return process.returncode
    except KeyboardInterrupt:
        print("\033[93mApplication stopped by user.\033[0m")
        return 0
    except Exception as e:
        print(f"\033[91mError running application: {e}\033[0m")
        return 1

if __name__ == "__main__":
    sys.exit(main())