#!/usr/bin/env python3
import subprocess
import shutil
import sys

def check_requirements():
    """Check if all requirements are installed."""
    print("🔍 Checking requirements...")
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check if Ollama is installed
    ollama_installed = shutil.which("ollama") is not None
    print(f"Ollama installed: {ollama_installed}")
    
    if not ollama_installed:
        print("❌ Ollama is not installed. Please install from https://ollama.ai")
        return False
    
    # Check if Ollama server is running
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Ollama server is running")
            return True
        else:
            print("❌ Ollama server is not running")
            return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False

if __name__ == "__main__":
    if check_requirements():
        print("\n✅ All requirements are satisfied!")
        print("You can now run the AI system with: python app.py")
    else:
        print("\n❌ Some requirements are missing.")
        sys.exit(1)
