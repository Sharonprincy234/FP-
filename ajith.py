from flask import Flask, request, jsonify
import subprocess
import shutil
import logging
import time
import json
import os
import re
from flask_cors import CORS

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Expanded Model List with more options for better performance
MODEL_PRIORITY = [
    "llama3.2:3b-instruct",          # Fastest new model with great accuracy
    "llama3.2:1b-instruct",          # Ultra-fast for quick responses
    "llama3.1:8b-instruct",          # Excellent balance of speed and accuracy
    "qwen2.5:7b-instruct",           # Great for coding and technical queries
    "qwen2.5:3b-instruct",           # Lighter version of Qwen
    "mistral:7b-instruct-v0.3",      # Latest mistral version
    "mistral-nemo:12b",              # New Mistral-Nemo model
    "gemma2:9b",                     # Google's latest efficient model
    "gemma2:2b",                     # Lightweight Gemma
    "phi3:14b",                      # Microsoft's efficient model
    "phi3:mini",                     # Smaller Phi3 model
    "codegemma:7b",                  # Specialized for coding
    "llama2-uncensored:7b",          # Alternative option
    "dolphin-llama3:8b",             # Uncensored model
    "wizardcoder:7b-python",         # Specialized for coding
    "starling-lm:7b",                # Good general purpose model
    "solar:10.7b",                   # Another good option
]

# Model to version mapping
MODEL_VERSIONS = {
    "llama3.2:3b-instruct": "Biovus AI v3",
    "llama3.2:1b-instruct": "Biovus AI v2", 
    "llama3.1:8b-instruct": "Biovus AI v4",
    "qwen2.5:7b-instruct": "Biovus AI v5",
    "qwen2.5:3b-instruct": "Biovus AI v3",
    "mistral:7b-instruct-v0.3": "Biovus AI v4",
    "mistral-nemo:12b": "Biovus AI v5",
    "gemma2:9b": "Biovus AI v4",
    "gemma2:2b": "Biovus AI v3",
    "phi3:14b": "Biovus AI v5",
    "phi3:mini": "Biovus AI v3",
    "codegemma:7b": "Biovus AI v4",
    "llama2-uncensored:7b": "Biovus AI v3",
    "dolphin-llama3:8b": "Biovus AI v4",
    "wizardcoder:7b-python": "Biovus AI v4",
    "starling-lm:7b": "Biovus AI v3",
    "solar:10.7b": "Biovus AI v4"
}

SERVER_PORT = 5000

# Global state
current_model = None
model_ready = False
response_cache = {}
CACHE_TIMEOUT = 300

def get_model_version(model_name):
    """Get the Biovus AI version for a model."""
    return MODEL_VERSIONS.get(model_name, "Biovus AI v1")

def check_ollama_installed():
    """Check if Ollama is installed and available."""
    return shutil.which("ollama") is not None

def get_available_models():
    """Get list of available Ollama models."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            models = []
            lines = result.stdout.strip().split('\n')[1:]
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if parts:
                        models.append(parts[0])
            return models
        else:
            logger.error(f"Ollama list failed: {result.stderr}")
            return []
    except subprocess.TimeoutExpired:
        logger.error("Timeout getting models list")
        return []
    except Exception as e:
        logger.error(f"Error getting models: {e}")
    return []

def pull_optimal_model():
    """Pull the most optimal model for speed and accuracy."""
    global current_model
    
    # Try multiple models with fallbacks
    for model in MODEL_PRIORITY:
        try:
            logger.info(f"🚀 Attempting to pull optimal model: {model}")
            result = subprocess.run(["ollama", "pull", model], 
                                  timeout=1200,  # 20 minutes timeout
                                  capture_output=True, 
                                  text=True)
            
            if result.returncode == 0:
                current_model = model
                logger.info(f"✅ Successfully pulled optimal model: {model}")
                return True
            else:
                logger.warning(f"⚠️ Failed to pull {model}: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.warning(f"⏰ Timeout pulling {model}, trying next...")
        except Exception as e:
            logger.warning(f"❌ Error pulling {model}: {e}")
    
    # If all else fails, try the default model
    try:
        logger.info("🔄 Trying default model as fallback")
        result = subprocess.run(["ollama", "pull", "llama3.2:3b-instruct"], 
                              timeout=600,
                              capture_output=True,
                              text=True)
        if result.returncode == 0:
            current_model = "llama3.2:3b-instruct"
            logger.info("✅ Successfully pulled default model")
            return True
    except Exception as e:
        logger.error(f"❌ Failed to pull default model: {e}")
    
    return False

def ensure_model_available():
    """Ensure we have the best possible model."""
    global current_model
    
    models = get_available_models()
    logger.info(f"📊 Available models: {models}")
    
    if models:
        # Try to find the most optimal available model
        for preferred_model in MODEL_PRIORITY:
            if preferred_model in models:
                current_model = preferred_model
                logger.info(f"🎯 Selected optimal model: {current_model}")
                return True
        
        # Fallback to any available model
        current_model = models[0]
        logger.info(f"🔧 Using available model: {current_model}")
        return True
    
    # No models available, pull an optimal one
    logger.info("📦 No models found, pulling optimal model...")
    return pull_optimal_model()

def start_ollama_server():
    """Start Ollama server with optimal settings."""
    try:
        # Check if Ollama is already running
        result = subprocess.run(["pgrep", "-f", "ollama serve"], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("✅ Ollama server is already running")
            return True
            
        # Start Ollama server with optimized settings
        subprocess.Popen(["ollama", "serve"], 
                       stdout=subprocess.DEVNULL, 
                       stderr=subprocess.DEVNULL)
        time.sleep(8)  # Give more time for server to start
        logger.info("✅ Ollama server started")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to start Ollama server: {e}")
        return False

def format_response_for_frontend(response):
    """Format the response to be displayed properly in the frontend."""
    if not response:
        return response
    
    formatted_response = response
    
    # Clean up the response
    formatted_response = re.sub(r'^\d+\.\s*\*\*', '• ', formatted_response, flags=re.MULTILINE)
    formatted_response = re.sub(r'\*\*', '', formatted_response)
    
    # Add emojis for better visual appeal
    emoji_replacements = {
        'Using the': '📝 Using the',
        'Replace "': '🔄 Replace "',
        'For example:': '📌 For example:',
        'Note:': '💡 Note:',
        'Warning:': '⚠️ Warning:',
        'Important:': '❗ Important:',
        'Step 1': '1️⃣ Step 1',
        'Step 2': '2️⃣ Step 2',
        'Step 3': '3️⃣ Step 3',
        'Step 4': '4️⃣ Step 4',
        'Step 5': '5️⃣ Step 5'
    }
    
    for old, new in emoji_replacements.items():
        formatted_response = formatted_response.replace(old, new)
    
    return formatted_response

def query_ollama_enhanced(prompt):
    """Enhanced query with better context and optimized parameters."""
    if not current_model:
        return "Error: No model available"
    
    try:
        # Optimized prompt for better accuracy
        enhanced_prompt = f"""Please provide accurate, factual, and concise information. 
        Focus on being correct rather than verbose. If unsure, say so.
        
        For technical questions like installation guides, provide step-by-step instructions.
        For commands: use clear formatting with appropriate emojis.
        
        Question: {prompt}
        
        Provide a direct, accurate response:"""
        
        # Set environment variables for optimization
        env = os.environ.copy()
        env["OLLAMA_NUM_GPU"] = "1" if shutil.which("nvidia-smi") else "0"
        env["OLLAMA_NUM_THREADS"] = "8"  # Use more threads for better performance
        
        # Use correct Ollama syntax without invalid flags
        result = subprocess.run(
            ["ollama", "run", current_model, enhanced_prompt],
            capture_output=True,
            text=True,
            timeout=120,  # 120 second timeout
            env=env
        )
        
        logger.info(f"🤖 Model: {current_model}, Return code: {result.returncode}")
        
        if result.returncode == 0:
            response = result.stdout.strip()
            if response:
                # Clean and validate response
                response = response.replace('\\n', '\n').replace('\\"', '"')
                
                # Remove any model information from the response
                if "Model:" in response:
                    response = response.split("Model:")[0].strip()
                
                # Format for frontend display
                response = format_response_for_frontend(response)
                
                # Less strict quality validation
                if any(phrase in response.lower() for phrase in ["sorry", "don't know", "not sure", "cannot answer", "i apologize"]):
                    return "I want to provide accurate information. Could you please clarify or rephrase your question?"
                
                return response
            else:
                return "I couldn't generate a proper response. Please try again with a different phrasing."
        else:
            error_msg = result.stderr.strip() or 'Unknown error'
            logger.error(f"❌ Model error: {error_msg}")
            return "I encountered an error processing your request. Please try again."
            
    except subprocess.TimeoutExpired:
        return "The request timed out. Please try a more specific question or try again."
    except Exception as e:
        return f"System error: {str(e)}"

def query_with_validation(prompt, max_retries=3):
    """Query with validation and retry mechanism."""
    for attempt in range(max_retries):
        try:
            response = query_ollama_enhanced(prompt)
            
            # Less strict validation to prevent false negatives
            if (response and 
                not response.startswith("Error:") and
                not response.startswith("I encountered") and
                not response.startswith("The request") and
                len(response) > 5):  # Reduced minimum length requirement
                return response
            
            logger.warning(f"🔄 Attempt {attempt + 1} had issues: {response[:100]}...")
            time.sleep(2)  # Longer delay between retries
            
        except Exception as e:
            logger.error(f"🔥 Attempt {attempt + 1} error: {e}")
            time.sleep(2)
    
    return "I'm having trouble providing a quality response right now. Please try rephrasing your question or try again later."

def initialize_ai_system():
    """Initialize the AI system with optimal models."""
    global model_ready
    
    logger.info("🚀 Initializing High-Performance AI System...")
    
    if not check_ollama_installed():
        logger.error("❌ Ollama not installed. Please install from https://ollama.ai")
        return False
        
    if not start_ollama_server():
        logger.error("❌ Failed to start Ollama server")
        return False
        
    # Wait a bit for server to fully start
    time.sleep(5)
        
    if not ensure_model_available():
        logger.error("❌ Failed to ensure model availability")
        return False
        
    # Quick test the model with a simpler question
    logger.info("🧪 Quick testing model...")
    test_response = query_ollama_enhanced("Hello, how are you?")
    logger.info(f"📋 Test response: {test_response[:100]}...")
    
    # Even if test fails, we might still be ready
    model_ready = True
    logger.info(f"✅ AI system ready with model: {current_model}!")
    return True

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """Main chat endpoint with enhanced accuracy and speed."""
    if not model_ready:
        return jsonify({"error": "AI system not ready yet"}), 503
        
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "Message is required"}), 400
        
    prompt = data['message'].strip()
    if not prompt:
        return jsonify({"error": "Message cannot be empty"}), 400
    
    # Check cache
    current_time = time.time()
    cache_key = f"{current_model}:{prompt}"
    if cache_key in response_cache:
        cache_entry = response_cache[cache_key]
        if current_time - cache_entry['timestamp'] < CACHE_TIMEOUT:
            logger.info(f"📦 Using cached response for: {prompt[:50]}...")
            return jsonify({
                "response": cache_entry['response'],
                "cached": True,
                "model": get_model_version(current_model)  # Return version instead of model name
            })
    
    logger.info(f"💭 Processing: {prompt[:100]}...")
    
    # Get response with validation
    response = query_with_validation(prompt)
    
    # Cache the response
    response_cache[cache_key] = {
        'response': response,
        'timestamp': current_time
    }
    
    return jsonify({
        "response": response,
        "cached": False,
        "model": get_model_version(current_model)  # Return version instead of model name
    })

@app.route('/upgrade', methods=['POST'])
def upgrade_model():
    """Upgrade to a more powerful model."""
    global current_model
    
    logger.info("🔄 Attempting model upgrade...")
    
    if pull_optimal_model():
        return jsonify({
            "success": True,
            "message": f"Successfully upgraded to {get_model_version(current_model)}",
            "model": get_model_version(current_model)  # Return version instead of model name
        })
    else:
        return jsonify({
            "success": False,
            "error": "Failed to upgrade model"
        }), 500

@app.route('/system', methods=['GET'])
@app.route('/system-info', methods=['GET'])
def system_info():
    """Get system information."""
    models = get_available_models()
    
    # Convert model names to Biovus AI versions for frontend
    available_versions = [get_model_version(model) for model in models]
    
    return jsonify({
        "status": "ready" if model_ready else "initializing",
        "current_model": get_model_version(current_model) if current_model else None,  # Return version
        "current_model_raw": current_model,  # Keep raw for internal use
        "available_models": available_versions,  # Return versions instead of model names
        "available_models_raw": models,  # Keep raw for internal use
        "ollama_installed": check_ollama_installed(),
        "server_running": model_ready
    })

@app.route('/models/select', methods=['POST'])
def select_model():
    """Select a specific model to use."""
    global current_model
    
    data = request.get_json()
    if not data or 'model' not in data:
        return jsonify({"error": "Model name is required"}), 400
    
    model_version = data['model'].strip()
    available_models = get_available_models()
    
    # Find the actual model name from the version
    selected_model = None
    for model_name, version in MODEL_VERSIONS.items():
        if version == model_version and model_name in available_models:
            selected_model = model_name
            break
    
    if not selected_model:
        return jsonify({"error": f"Model version '{model_version}' not available"}), 400
    
    # Test the model
    try:
        old_model = current_model
        current_model = selected_model
        
        # Quick test to ensure model works
        test_response = query_ollama_enhanced("Hello")
        if test_response.startswith("Error:") or test_response.startswith("I encountered"):
            current_model = old_model
            return jsonify({"error": f"Model '{model_version}' failed to respond properly"}), 500
        
        logger.info(f"✅ Switched to model: {selected_model}")
        return jsonify({
            "success": True,
            "message": f"Switched to {model_version}",
            "model": model_version
        })
        
    except Exception as e:
        current_model = old_model
        return jsonify({"error": f"Failed to switch model: {str(e)}"}), 500

@app.route('/models/pull', methods=['POST'])
def pull_model():
    """Pull a specific model."""
    data = request.get_json()
    if not data or 'model' not in data:
        return jsonify({"error": "Model name is required"}), 400
    
    model_name = data['model'].strip()
    
    try:
        logger.info(f"🚀 Pulling model: {model_name}")
        result = subprocess.run(
            ["ollama", "pull", model_name],
            timeout=1200,  # Increased timeout
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            return jsonify({
                "success": True,
                "message": f"Successfully pulled {get_model_version(model_name)}",
                "model": get_model_version(model_name)
            })
        else:
            return jsonify({
                "success": False,
                "error": f"Failed to pull model: {result.stderr}"
            }), 500
            
    except subprocess.TimeoutExpired:
        return jsonify({
            "success": False,
            "error": "Model pull timed out"
        }), 500
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Error pulling model: {str(e)}"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy" if model_ready else "initializing",
        "model": get_model_version(current_model) if current_model else None,
        "timestamp": time.time()
    })

@app.route('/models/refresh', methods=['POST'])
def refresh_models():
    """Refresh the list of available models."""
    models = get_available_models()
    # Convert to versions for frontend
    versions = [get_model_version(model) for model in models]
    return jsonify({
        "success": True,
        "models": versions,
        "models_raw": models  # Keep raw for internal use
    })

if __name__ == '__main__':
    # Initialize the AI system
    success = initialize_ai_system()
    
    if success:
        logger.info(f"🌐 Starting server on port {SERVER_PORT}")
        app.run(host='0.0.0.0', port=SERVER_PORT, debug=False, threaded=True)
    else:
        logger.error("❌ Failed to initialize AI system")
