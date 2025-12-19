# app.py - Complete AI system with optimized model selection
from flask import Flask, request, jsonify
import logging
import time
import shutil
import subprocess
import os
import uuid
import re
from datetime import datetime
from flask_cors import CORS

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24).hex()
CORS(app)

# Model configuration
MODEL_CONFIG = {
    "general_qa": {
        "models": [
            "llama3.2:3b-instruct",
            "phi3:mini",
            "qwen2.5:3b-instruct"
        ],
        "description": "General knowledge and question answering",
        "priority": 1
    },
    "programming": {
        "models": [
            "codegemma:7b",
            "phi3:mini",
            "llama3.2:3b-instruct"
        ],
        "description": "Coding, programming, and technical questions",
        "priority": 2
    },
    "technical_os": {
        "models": [
            "llama3.2:3b-instruct",
            "phi3:mini",
            "qwen2.5:3b-instruct"
        ],
        "description": "Linux, Windows, and OS-related questions",
        "priority": 3
    },
    "emotional_analysis": {
        "models": [
            "llama3.2:3b-instruct",
            "phi3:mini",
            "qwen2.5:3b-instruct"
        ],
        "description": "Emotional analysis and sentiment detection",
        "priority": 4
    },
    "creative_writing": {
        "models": [
            "llama3.2:3b-instruct",
            "phi3:mini",
            "qwen2.5:3b-instruct"
        ],
        "description": "Creative writing and storytelling",
        "priority": 5
    },
    "research_deep_dive": {
        "models": [
            "llama3.2:3b-instruct",
            "phi3:mini",
            "qwen2.5:3b-instruct"
        ],
        "description": "Research and in-depth analysis",
        "priority": 6
    },
    "fast_response": {
        "models": [
            "llama3.2:1b-instruct",
            "phi3:mini",
            "qwen2.5:0.5b"
        ],
        "description": "Quick responses for simple queries",
        "priority": 7
    }
}

MODEL_VERSIONS = {
    "llama3.2:3b-instruct": "Biovus AI Pro v2023",
    "llama3.2:1b-instruct": "Biovus AI Lite v2023",
    "phi3:mini": "Biovus AI Fast v2023",
    "qwen2.5:3b-instruct": "Biovus AI Multilingual v2023",
    "qwen2.5:0.5b": "Biovus AI Micro v2023",
    "codegemma:7b": "Biovus AI Coder v2023"
}

SERVER_PORT = 5000
CACHE_TIMEOUT = 300

# Response cache
class ResponseCache:
    def __init__(self, timeout=300):
        self.cache = {}
        self.timeout = timeout
        
    def get(self, key):
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['timestamp'] < self.timeout:
                return entry['value']
            else:
                del self.cache[key]
        return None
        
    def set(self, key, value):
        self.cache[key] = {
            'value': value,
            'timestamp': time.time()
        }
        
    def clear(self):
        self.cache.clear()

# Query analyzer
class QueryAnalyzer:
    def __init__(self):
        self.programming_keywords = [
            'code', 'programming', 'python', 'javascript', 'java', 'c++', 'html', 'css',
            'function', 'algorithm', 'variable', 'loop', 'debug', 'compile', 'syntax',
            'api', 'framework', 'library', 'git', 'github', 'docker', 'kubernetes',
            'bug', 'error', 'fix', 'develop', 'software', 'app', 'application'
        ]
        
        self.technical_os_keywords = [
            'linux', 'ubuntu', 'debian', 'centos', 'windows', 'macos', 'command',
            'terminal', 'bash', 'shell', 'install', 'update', 'upgrade', 'package',
            'server', 'nginx', 'apache', 'database', 'mysql', 'postgresql', 'ssh',
            'permission', 'file system', 'directory', 'process', 'kernel', 'os',
            'ubuntu', 'centos', 'debian', 'redhat', 'fedora'
        ]
        
        self.emotional_keywords = [
            'feel', 'feeling', 'emotion', 'emotional', 'sad', 'happy', 'angry',
            'anxious', 'depressed', 'stress', 'stressed', 'worried', 'nervous',
            'excited', 'joy', 'love', 'hate', 'relationship', 'friend', 'family'
        ]
        
        self.creative_keywords = [
            'story', 'poem', 'creative', 'write a', 'imagine', 'fiction', 'narrative',
            'character', 'plot', 'setting', 'describe', 'metaphor', 'simile'
        ]
        
        self.research_keywords = [
            'research', 'study', 'analysis', 'analyze', 'compare', 'contrast',
            'statistics', 'data', 'survey', 'experiment', 'theory', 'hypothesis'
        ]

    def analyze_query(self, query):
        query_lower = query.lower()
        
        category_scores = {
            'programming': self._calculate_score(query_lower, self.programming_keywords),
            'technical_os': self._calculate_score(query_lower, self.technical_os_keywords),
            'emotional_analysis': self._calculate_score(query_lower, self.emotional_keywords),
            'creative_writing': self._calculate_score(query_lower, self.creative_keywords),
            'research_deep_dive': self._calculate_score(query_lower, self.research_keywords),
        }
        
        word_count = len(query_lower.split())
        if word_count <= 5:
            category_scores['fast_response'] = 0.8
        else:
            category_scores['fast_response'] = 0.1
        
        best_category = max(category_scores.items(), key=lambda x: x[1])
        
        if best_category[1] < 0.3:
            best_category = ('general_qa', 0.5)
        
        logger.info(f"Query: '{query}' → Category: {best_category[0]} (Score: {best_category[1]:.2f})")
        
        return best_category[0], {
            'category': best_category[0],
            'confidence': best_category[1],
            'scores': category_scores
        }

    def _calculate_score(self, query, keywords):
        score = 0
        total_keywords = len(keywords)
        
        for keyword in keywords:
            if keyword in query:
                score += 1
                if f" {keyword} " in f" {query} ":
                    score += 0.5
        
        if total_keywords > 0:
            normalized_score = min(score / (total_keywords * 0.3), 1.0)
        else:
            normalized_score = 0
            
        return normalized_score

# Response formatter
class ResponseFormatter:
    def __init__(self):
        self.pattern_replacements = [
            (r'(?i)\bnote:\b', '📝 Note:'),
            (r'(?i)\btip:\b', '💡 Tip:'),
            (r'(?i)\bwarning:\b', '⚠️ Warning:'),
            (r'(?i)\bimportant:\b', '❗ Important:'),
            (r'(?i)\bexample:\b', '📌 Example:'),
            (r'(?i)\bstep\s+(\d+):', r'🔹 Step \1:'),
            (r'^- ', '• '),
            (r'^\d+\.', '•')
        ]
        
    def format_response(self, response, query_type):
        if not response:
            return response
            
        formatted = response.strip()
        formatted = re.sub(r'(?i)(model:|assistant:|system:).*$', '', formatted, flags=re.MULTILINE)
        
        for pattern, replacement in self.pattern_replacements:
            formatted = re.sub(pattern, replacement, formatted, flags=re.MULTILINE)
        
        if query_type == 'programming':
            formatted = self._format_code_response(formatted)
        elif query_type == 'technical_os':
            formatted = self._format_tech_response(formatted)
        
        formatted = re.sub(r'\n{3,}', '\n\n', formatted)
        
        return formatted
        
    def _format_code_response(self, response):
        lines = response.split('\n')
        in_code_block = False
        code_block_lines = []
        formatted_lines = []
        
        for line in lines:
            if line.strip().startswith('```'):
                if in_code_block and code_block_lines:
                    if len(code_block_lines) > 5:
                        numbered_lines = []
                        for i, code_line in enumerate(code_block_lines, 1):
                            numbered_lines.append(f"{i:2d} | {code_line}")
                        formatted_lines.extend(numbered_lines)
                    else:
                        formatted_lines.extend(code_block_lines)
                    code_block_lines = []
                
                in_code_block = not in_code_block
                formatted_lines.append(line)
            elif in_code_block:
                code_block_lines.append(line)
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
        
    def _format_tech_response(self, response):
        tech_patterns = [
            (r'(?i)command:', '💻 Command:'),
            (r'(?i)step\s*\d+:', '🔹 Step:'),
            (r'(?i)solution:', '✅ Solution:'),
            (r'(?i)error:', '❌ Error:')
        ]
        
        for pattern, replacement in tech_patterns:
            response = re.sub(pattern, replacement, response)
            
        return response

# Model manager
class ModelManager:
    def __init__(self):
        self.current_model = None
        self.available_models = []
        self.model_ready = False
        
    def initialize(self):
        logger.info("🚀 Initializing Model Manager...")
        
        if not self._check_ollama_installed():
            logger.error("❌ Ollama not installed. Please install from https://ollama.ai")
            return False
            
        if not self._check_ollama_running():
            logger.error("❌ Ollama server not running. Please start it with: ollama serve")
            return False
            
        self.available_models = self._get_available_models()
        
        if not self.available_models:
            logger.warning("⚠️ No models available. Using fallback mode")
        else:
            logger.info(f"✅ Found {len(self.available_models)} models: {', '.join(self.available_models)}")
            self.current_model = self.available_models[0]
            
        self.model_ready = True
        logger.info("✅ Model Manager initialized")
        return True
        
    def _check_ollama_installed(self):
        return shutil.which("ollama") is not None
        
    def _check_ollama_running(self):
        try:
            result = subprocess.run(["curl", "-s", "http://localhost:11434/api/tags"], 
                                 capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False
        
    def _get_available_models(self):
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                models = []
                lines = result.stdout.strip().split('\n')
                for line in lines[1:]:
                    if line.strip():
                        parts = line.split()
                        if parts:
                            model_name = parts[0]
                            models.append(model_name)
                return models
            return []
        except:
            return []
        
    def get_best_model_for_category(self, category):
        if not self.available_models:
            return None
            
        if category not in MODEL_CONFIG:
            return self.available_models[0]
            
        for model in MODEL_CONFIG[category]['models']:
            if model in self.available_models:
                return model
                
        return self.available_models[0]
        
    def get_model_version(self, model_name):
        return MODEL_VERSIONS.get(model_name, f"Biovus AI ({model_name})")
        
    def is_ready(self):
        return self.model_ready

# Initialize components
model_manager = ModelManager()
query_analyzer = QueryAnalyzer()
response_formatter = ResponseFormatter()
response_cache = ResponseCache()

# Session management
class SimpleSessionManager:
    def __init__(self):
        self.sessions = {}
    
    def get_session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'chat_id': str(uuid.uuid4()),
                'chat_history': [],
                'model_preference': None,
                'created': time.time()
            }
        return self.sessions[session_id]
    
    def cleanup_old_sessions(self, timeout=3600):
        current_time = time.time()
        for session_id in list(self.sessions.keys()):
            if current_time - self.sessions[session_id]['created'] > timeout:
                del self.sessions[session_id]

session_manager = SimpleSessionManager()

@app.before_request
def before_request():
    session_id = request.cookies.get('session_id') or str(uuid.uuid4())
    request.session = session_manager.get_session(session_id)

@app.after_request
def after_request(response):
    if 'session_id' not in request.cookies:
        response.set_cookie('session_id', request.session['chat_id'], max_age=3600)
    return response

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    if not model_manager.is_ready():
        return jsonify({
            "error": "AI system not ready yet",
            "type": "system_error",
            "timestamp": datetime.now().isoformat()
        }), 503
        
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({
            "error": "Message is required",
            "type": "validation_error",
            "timestamp": datetime.now().isoformat()
        }), 400
        
    prompt = data['message'].strip()
    if not prompt:
        return jsonify({
            "error": "Message cannot be empty",
            "type": "validation_error",
            "timestamp": datetime.now().isoformat()
        }), 400
    
    chat_history = request.session.get('chat_history', [])
    context = build_context(chat_history[-5:]) if chat_history else ""
    
    cache_key = f"{request.session['chat_id']}:{prompt}:{hash(context)}"
    cached_response = response_cache.get(cache_key)
    if cached_response:
        logger.info(f"📦 Using cached response for: {prompt[:50]}...")
        
        chat_entry = {
            "id": str(uuid.uuid4()),
            "type": "assistant",
            "content": cached_response['response'],
            "timestamp": datetime.now().isoformat(),
            "model": cached_response['model'],
            "category": cached_response['category'],
            "cached": True
        }
        request.session['chat_history'].append(chat_entry)
        
        return jsonify({
            "response": cached_response['response'],
            "message_id": chat_entry['id'],
            "cached": True,
            "model": cached_response['model'],
            "category": cached_response['category'],
            "timestamp": chat_entry['timestamp'],
            "conversation_id": request.session['chat_id']
        })
    
    logger.info(f"💭 Processing: {prompt[:100]}...")
    
    user_entry = {
        "id": str(uuid.uuid4()),
        "type": "user",
        "content": prompt,
        "timestamp": datetime.now().isoformat()
    }
    request.session['chat_history'].append(user_entry)
    
    category, analysis = query_analyzer.analyze_query(prompt)
    
    preferred_model = request.session.get('model_preference')
    if preferred_model and preferred_model in model_manager.available_models:
        model_name = preferred_model
    else:
        model_name = model_manager.get_best_model_for_category(category)
    
    if not model_name:
        return jsonify({
            "error": f"No suitable model available for {category} queries",
            "type": "model_error",
            "timestamp": datetime.now().isoformat()
        }), 503
    
    response = query_model_with_context(model_name, prompt, category, context)
    
    formatted_response = response_formatter.format_response(response, category)
    
    assistant_entry = {
        "id": str(uuid.uuid4()),
        "type": "assistant",
        "content": formatted_response,
        "timestamp": datetime.now().isoformat(),
        "model": model_manager.get_model_version(model_name),
        "category": category,
        "cached": False
    }
    request.session['chat_history'].append(assistant_entry)
    
    response_cache.set(cache_key, {
        'response': formatted_response,
        'model': model_manager.get_model_version(model_name),
        'category': category
    })
    
    return jsonify({
        "response": formatted_response,
        "message_id": assistant_entry['id'],
        "cached": False,
        "model": model_manager.get_model_version(model_name),
        "category": category,
        "analysis": analysis,
        "timestamp": assistant_entry['timestamp'],
        "conversation_id": request.session['chat_id']
    })

def build_context(recent_messages):
    context = ""
    for msg in recent_messages:
        if msg['type'] == 'user':
            context += f"User: {msg['content']}\n"
        else:
            context += f"Assistant: {msg['content']}\n"
    return context

def query_model_with_context(model_name, prompt, category, context):
    try:
        enhanced_prompt = create_contextual_prompt(prompt, category, context)
        
        env = os.environ.copy()
        env["OLLAMA_NUM_GPU"] = "1" if shutil.which("nvidia-smi") else "0"
        env["OLLAMA_NUM_THREADS"] = "8"
        
        result = subprocess.run(
            ["ollama", "run", model_name, enhanced_prompt],
            capture_output=True,
            text=True,
            timeout=120,
            env=env
        )
        
        if result.returncode == 0:
            response = result.stdout.strip()
            if response:
                response = response.replace('\\n', '\n').replace('\\"', '"')
                
                lines = response.split('\n')
                cleaned_lines = []
                for line in lines:
                    if not any(x in line for x in ['Model:', 'User:', 'Assistant:', 'System:']):
                        cleaned_lines.append(line)
                
                response = '\n'.join(cleaned_lines).strip()
                return response if response else "I'll need more context to provide a helpful response."
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

def create_contextual_prompt(prompt, category, context):
    base_instructions = """You are Biovus AI, an advanced AI assistant. Provide accurate, helpful responses.

Current conversation context:
{context}

New message from user: {prompt}

Please provide a thoughtful, context-aware response:"""
    
    category_specific = {
        "general_qa": "Provide a clear, comprehensive answer to the question:",
        "programming": "Provide accurate code with explanations. Format code properly:",
        "technical_os": "Provide step-by-step technical guidance. Be detailed but clear:",
        "emotional_analysis": "Provide empathetic, understanding support:",
        "creative_writing": "Create engaging, creative content:",
        "research_deep_dive": "Provide in-depth, well-researched analysis:",
        "fast_response": "Provide a quick, concise answer:"
    }
    
    category_instruction = category_specific.get(category, "Respond appropriately to:")
    
    return base_instructions.format(context=context, prompt=prompt) + f"\n\n{category_instruction}"

@app.route('/conversation', methods=['GET'])
def get_conversation():
    return jsonify({
        "conversation_id": request.session.get('chat_id'),
        "messages": request.session.get('chat_history', []),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/conversation/clear', methods=['POST'])
def clear_conversation():
    request.session['chat_history'] = []
    request.session['chat_id'] = str(uuid.uuid4())
    
    return jsonify({
        "success": True,
        "message": "Conversation cleared",
        "new_conversation_id": request.session['chat_id'],
        "timestamp": datetime.now().isoformat()
    })

@app.route('/system-info', methods=['GET'])
def system_info():
    return jsonify({
        "status": "ready" if model_manager.is_ready() else "initializing",
        "current_model": model_manager.get_model_version(model_manager.current_model) if model_manager.current_model else None,
        "available_models": model_manager.available_models,
        "model_categories": list(MODEL_CONFIG.keys()),
        "conversation_id": request.session.get('chat_id'),
        "message_count": len(request.session.get('chat_history', [])),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/models/select', methods=['POST'])
def select_model():
    data = request.get_json()
    if not data or 'model' not in data:
        return jsonify({
            "error": "Model name is required",
            "type": "validation_error",
            "timestamp": datetime.now().isoformat()
        }), 400
    
    model_name = data['model'].strip()
    
    if model_name not in model_manager.available_models:
        return jsonify({
            "error": f"Model '{model_name}' not available",
            "type": "model_error",
            "timestamp": datetime.now().isoformat()
        }), 400
    
    request.session['model_preference'] = model_name
    
    return jsonify({
        "success": True,
        "message": f"Model preference set to {model_manager.get_model_version(model_name)}",
        "model": model_manager.get_model_version(model_name),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy" if model_manager.is_ready() else "initializing",
        "model": model_manager.get_model_version(model_manager.current_model) if model_manager.current_model else None,
        "conversation_id": request.session.get('chat_id'),
        "timestamp": datetime.now().isoformat(),
        "uptime": time.time() - app_start_time
    })

app_start_time = time.time()

if __name__ == '__main__':
    success = model_manager.initialize()
    
    if success:
        logger.info(f"🌐 Starting server on port {SERVER_PORT}")
        app.run(host='0.0.0.0', port=SERVER_PORT, debug=False, threaded=True)
    else:
        logger.error("❌ Failed to initialize AI system")
