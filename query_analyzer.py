import re
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class QueryAnalyzer:
    """Analyzes queries to determine the most appropriate model category."""
    
    def __init__(self):
        # Keywords for different query types
        self.programming_keywords = [
            'code', 'programming', 'python', 'javascript', 'java', 'c++', 'html', 'css',
            'function', 'algorithm', 'variable', 'loop', 'debug', 'compile', 'syntax',
            'api', 'framework', 'library', 'git', 'github', 'docker', 'kubernetes'
        ]
        
        self.technical_os_keywords = [
            'linux', 'ubuntu', 'debian', 'centos', 'windows', 'macos', 'command',
            'terminal', 'bash', 'shell', 'install', 'update', 'upgrade', 'package',
            'server', 'nginx', 'apache', 'database', 'mysql', 'postgresql', 'ssh',
            'permission', 'file system', 'directory', 'process', 'kernel'
        ]
        
        self.emotional_keywords = [
            'feel', 'feeling', 'emotion', 'emotional', 'sad', 'happy', 'angry',
            'anxious', 'depressed', 'stress', 'stressed', 'worried', 'nervous',
            'excited', 'joy', 'love', 'hate', 'relationship', 'friend', 'family',
            'therapy', 'counseling', 'mental health', 'psychology'
        ]
        
        self.creative_keywords = [
            'story', 'poem', 'creative', 'write a', 'imagine', 'fiction', 'narrative',
            'character', 'plot', 'setting', 'describe', 'metaphor', 'simile',
            'fantasy', 'sci-fi', 'science fiction', 'romance', 'mystery', 'thriller'
        ]
        
        self.research_keywords = [
            'research', 'study', 'analysis', 'analyze', 'compare', 'contrast',
            'statistics', 'data', 'survey', 'experiment', 'theory', 'hypothesis',
            'academic', 'scholarly', 'journal', 'paper', 'thesis', 'dissertation'
        ]

    def analyze_query(self, query: str) -> Tuple[str, Dict]:
        """Analyze the query and return the most appropriate category and confidence."""
        query_lower = query.lower()
        
        # Check for category matches
        category_scores = {
            'programming': self._calculate_score(query_lower, self.programming_keywords),
            'technical_os': self._calculate_score(query_lower, self.technical_os_keywords),
            'emotional_analysis': self._calculate_score(query_lower, self.emotional_keywords),
            'creative_writing': self._calculate_score(query_lower, self.creative_keywords),
            'research_deep_dive': self._calculate_score(query_lower, self.research_keywords),
        }
        
        # Add length-based scoring for fast_response
        word_count = len(query_lower.split())
        if word_count <= 5:
            category_scores['fast_response'] = 0.8
        else:
            category_scores['fast_response'] = 0.1
        
        # Find the category with the highest score
        best_category = max(category_scores.items(), key=lambda x: x[1])
        
        # If no strong match, use general_qa
        if best_category[1] < 0.3:
            best_category = ('general_qa', 0.5)
        
        logger.info(f"Query: '{query}' → Category: {best_category[0]} (Score: {best_category[1]:.2f})")
        
        return best_category[0], {
            'category': best_category[0],
            'confidence': best_category[1],
            'scores': category_scores
        }

    def _calculate_score(self, query: str, keywords: List[str]) -> float:
        """Calculate a score based on keyword matches."""
        score = 0
        total_keywords = len(keywords)
        
        for keyword in keywords:
            if keyword in query:
                score += 1
                
                # Bonus for exact matches
                if f" {keyword} " in f" {query} ":
                    score += 0.5
        
        # Normalize score
        if total_keywords > 0:
            normalized_score = min(score / (total_keywords * 0.3), 1.0)
        else:
            normalized_score = 0
            
        return normalized_score
