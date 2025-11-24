#!/bin/bash
# Quick setup script for AI Trading System

echo "🚀 AI Trading System Setup"
echo "=========================="
echo ""

# Check Python version
echo "✓ Checking Python version..."
python3 --version

# Create virtual environment
echo "✓ Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "✓ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "✓ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "✓ Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Test the system: python src/main.py scan"
echo "3. View logs: tail -f logs/trading.log"
echo ""
echo "📚 See README.md for full documentation"
