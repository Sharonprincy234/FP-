import subprocess
import time
import logging
from typing import Dict, List, Optional

from config.models import MODEL_CONFIG

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Registry for managing model information and status."""
    
    def __init__(self):
        self.models = {}
        self._initialize_model_registry()
        
    def _initialize_model_registry(self):
        """Initialize the model registry with all configured models."""
        for category, config in MODEL_CONFIG.items():
            for model_name in config['models']:
                self.models[model_name] = {
                    'name': model_name,
                    'category': category,
                    'description': config['description'],
                    'priority': config['priority'],
                    'status': 'unknown',
                    'last_used': None
                }
                
    def update_model_status(self, model_name: str, status: str):
        """Update the status of a model."""
        if model_name in self.models:
            self.models[model_name]['status'] = status
            self.models[model_name]['last_used'] = time.time()
            
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get information about a model."""
        return self.models.get(model_name)
        
    def get_models_by_category(self, category: str) -> List[Dict]:
        """Get all models for a specific category."""
        return [
            model for model in self.models.values() 
            if model['category'] == category
        ]
        
    def get_all_models(self) -> List[Dict]:
        """Get all models."""
        return list(self.models.values())
