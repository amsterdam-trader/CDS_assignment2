#!/bin/bash
# Climate Data Science Assignment 1 - Environment Setup Script (Unix/Mac)
# This script creates a virtual environment and installs all dependencies

echo "================================================================================"
echo "Climate Data Science Assignment 1 - Environment Setup"
echo "================================================================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

echo "[1/4] Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi

echo "[2/4] Activating virtual environment..."
source venv/bin/activate

echo "[3/4] Upgrading pip..."
python -m pip install --upgrade pip

echo "[4/4] Installing requirements..."
pip install -r requirements.txt

echo ""
echo "================================================================================"
echo "Setup complete!"
echo "================================================================================"
echo ""
echo "To activate the environment in the future, run:"
echo "    source venv/bin/activate"
echo ""
echo "To run the analyses:"
echo "    1. Activate the environment (see above)"
echo "    2. Open main.ipynb in Jupyter"
echo "    OR"
echo "    Run individual scripts: python -m src.q1"
echo ""
