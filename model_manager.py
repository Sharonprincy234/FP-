import subprocess
import shutil
import logging
import time
from typing import List, Dict, Optional

from config.models import MODEL_CONFIG, MODEL_VERSIONS

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages AI models, their availability, and selection."""
    
    def __init__(self):
        self.current_model = None
        self.available_models = []
        self.model_ready = False
        
    def initialize(self) -> bool:
        """Initialize the model system."""
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
            # Continue without models - will use fallback logic
        else:
            logger.info(f"✅ Found {len(self.available_models)} models: {', '.join(self.available_models)}")
            self.current_model = self.available_models[0]
            
        self.model_ready = True
        logger.info("✅ Model Manager initialized")
        return True
        
    def _check_ollama_installed(self) -> bool:
        """Check if Ollama is installed."""
        return shutil.which("ollama") is not None
        
    def _check_ollama_running(self) -> bool:
        """Check if Ollama server is running."""
        try:
            result = subprocess.run(["curl", "-s", "http://localhost:11434/api/tags"], 
                                 capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False
        
    def _get_available_models(self) -> List[str]:
        """Get list of available Ollama models."""
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                models = []
                lines = result.stdout.strip().split('\n')
                for line in lines[1:]:  # Skip header
                    if line.strip():
                        parts = line.split()
                        if parts:
                            model_name = parts[0]
                            models.append(model_name)
                return models
            return []
        except:
            return []
        
    def get_best_model_for_category(self, category: str) -> Optional[str]:
        """Get the best available model for a category."""
        if not self.available_models:
            return None
            
        if category not in MODEL_CONFIG:
            return self.available_models[0]  # Fallback to first available
            
        # Check which models are available for this category
        for model in MODEL_CONFIG[category]['models']:
            if model in self.available_models:
                return model
                
        # Fallback to any available model
        return self.available_models[0]
        
    def get_model_version(self, model_name: str) -> str:
        """Get the version name for a model."""
        return MODEL_VERSIONS.get(model_name, f"Biovus AI ({model_name})")
        
    def is_ready(self) -> bool:
        """Check if the model manager is ready."""
        return self.model_ready
