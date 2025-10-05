#!/bin/bash
# InvestmentIQ Virtual Environment Setup Script

echo "🚀 Setting up InvestmentIQ virtual environment..."

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies (this may take a while)..."
pip install -r requirements.txt

# Install project in editable mode
echo "📦 Installing project in editable mode..."
pip install -e .

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the dashboard:"
echo "  streamlit run apps/dashboard.py"
echo ""
