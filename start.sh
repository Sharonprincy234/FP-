#!/bin/bash

# Start the Advanced AI System

echo "🚀 Starting Advanced AI System..."

# Check if Python virtual environment exists
if [ ! -d "myenv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv myenv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source myenv/bin/activate

# Install requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt

# Test installation
echo "🧪 Testing installation..."
python test_installation.py

if [ $? -eq 0 ]; then
    echo "✅ Starting server..."
    python app.py
else
    echo "❌ Installation test failed. Please check the errors above."
    exit 1
fi
