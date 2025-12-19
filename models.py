# models.py - Updated with smaller models
# Model configuration with specialized models for different tasks

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

# Model to version mapping
MODEL_VERSIONS = {
    "llama3.2:3b-instruct": "Biovus AI Pro v2023",
    "llama3.2:1b-instruct": "Biovus AI Lite v2023",
    "phi3:mini": "Biovus AI Fast v2023",
    "qwen2.5:3b-instruct": "Biovus AI Multilingual v2023",
    "qwen2.5:0.5b": "Biovus AI Micro v2023",
    "codegemma:7b": "Biovus AI Coder v2023"
}

# Server configuration
SERVER_PORT = 5000
CACHE_TIMEOUT = 300
OLLAMA_HOST = "http://localhost:11434"
