# response_formatter.py - Enhanced with better formatting
import re
import html
from typing import Dict

class ResponseFormatter:
    """Formats responses for better presentation with advanced formatting."""
    
    def __init__(self):
        self.emoji_mapping = {
            'important': '❗',
            'warning': '⚠️',
            'tip': '💡',
            'note': '📝',
            'example': '📌',
            'success': '✅',
            'error': '❌',
            'info': 'ℹ️',
            'question': '❓',
            'idea': '💭',
            'solution': '🔧',
            'book': '📚',
            'link': '🔗',
            'download': '📥',
            'upload': '📤',
            'time': '⏰',
            'calendar': '📅',
            'location': '📍',
            'phone': '📞',
            'email': '📧',
            'heart': '❤️',
            'star': '⭐',
            'fire': '🔥',
            'thumbsup': '👍',
            'thumbsdown': '👎'
        }
        
        self.pattern_replacements = [
            (r'(?i)\bnote:\b', '📝 Note:'),
            (r'(?i)\btip:\b', '💡 Tip:'),
            (r'(?i)\bwarning:\b', '⚠️ Warning:'),
            (r'(?i)\bimportant:\b', '❗ Important:'),
            (r'(?i)\bexample:\b', '📌 Example:'),
            (r'(?i)\bstep\s+(\d+):', r'🔹 Step \1:'),
            (r'(?i)\bstep\s+(\d+)', r'🔹 Step \1'),
            (r'(?i)\bpros:\b', '✅ Pros:'),
            (r'(?i)\bcons:\b', '❌ Cons:'),
            (r'(?i)\badvantages:\b', '✅ Advantages:'),
            (r'(?i)\bdisadvantages:\b', '❌ Disadvantages:'),
            (r'^- ', '• '),
            (r'^\d+\.', '•')
        ]
        
    def format_response(self, response: str, query_type: str) -> str:
        """Format a response with advanced styling."""
        if not response:
            return response
            
        # Basic cleaning
        formatted = response.strip()
        
        # Remove any model signature lines
        formatted = re.sub(r'(?i)(model:|assistant:|system:).*$', '', formatted, flags=re.MULTILINE)
        
        # Apply pattern-based replacements
        for pattern, replacement in self.pattern_replacements:
            formatted = re.sub(pattern, replacement, formatted, flags=re.MULTILINE)
        
        # Type-specific formatting
        if query_type == 'programming':
            formatted = self._format_code_response(formatted)
        elif query_type == 'emotional_analysis':
            formatted = self._format_emotional_response(formatted)
        elif query_type == 'creative_writing':
            formatted = self._format_creative_response(formatted)
        elif query_type == 'research_deep_dive':
            formatted = self._format_research_response(formatted)
        
        # Ensure proper paragraph spacing
        formatted = re.sub(r'\n{3,}', '\n\n', formatted)
        
        return formatted
        
    def _format_code_response(self, response: str) -> str:
        """Format code-related responses with proper code blocks."""
        # Ensure code blocks are properly formatted
        response = re.sub(r'```(\w+)?\s*\n', r'```\1\n', response)
        response = re.sub(r'\n```', '\n```', response)
        
        # Add line numbers to large code blocks
        lines = response.split('\n')
        in_code_block = False
        code_block_lines = []
        formatted_lines = []
        
        for line in lines:
            if line.strip().startswith('```'):
                if in_code_block and code_block_lines:
                    # Process the completed code block
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
        
    def _format_emotional_response(self, response: str) -> str:
        """Format emotional responses with empathetic styling."""
        empathetic_patterns = [
            (r'(?i)i understand', '💭 I understand'),
            (r'(?i)i hear you', '👂 I hear you'),
            (r'(?i)that sounds', '🎭 That sounds'),
            (r'(?i)it seems', '🔍 It seems'),
            (r'(?i)you might be feeling', '❤️ You might be feeling'),
            (r'(?i)that must be', '💫 That must be'),
            (r'(?i)thank you for sharing', '🙏 Thank you for sharing')
        ]
        
        for pattern, replacement in empathetic_patterns:
            response = re.sub(pattern, replacement, response)
            
        return response
        
    def _format_creative_response(self, response: str) -> str:
        """Format creative writing responses."""
        # Add creative emojis and formatting
        creative_patterns = [
            (r'^(.*[.!?])$', r'✨ \1'),  # Add sparkle to sentence starts
            (r'\b(once upon a time|long ago|in a land)\b', r'🏰 \1'),
            (r'\b(magic|magical|enchant)\b', r'🔮 \1'),
            (r'\b(adventure|quest|journey)\b', r'🗺️ \1'),
            (r'\b(hero|heroine|champion)\b', r'🦸 \1'),
            (r'\b(villain|antagonist|evil)\b', r'🦹 \1')
        ]
        
        for pattern, replacement in creative_patterns:
            response = re.sub(pattern, replacement, response, flags=re.IGNORECASE)
            
        return response
        
    def _format_research_response(self, response: str) -> str:
        """Format research responses with academic styling."""
        research_patterns = [
            (r'\b(according to|studies show|research indicates)\b', r'📚 \1'),
            (r'\b(data suggests|evidence shows|analysis reveals)\b', r'📊 \1'),
            (r'\b(in conclusion|to summarize|overall)\b', r'📋 \1'),
            (r'\b(reference|citation|source)\b', r'📖 \1'),
            (r'\b(statistic|percentage|ratio)\b', r'📈 \1')
        ]
        
        for pattern, replacement in research_patterns:
            response = re.sub(pattern, replacement, response, flags=re.IGNORECASE)
            
        # Format lists with proper bullet points
        response = re.sub(r'^\d+\.\s+', '• ', response, flags=re.MULTILINE)
        
        return response
