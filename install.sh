#!/bin/bash

echo "========================================="
echo "  Manga Translator - Installation (Mac)"
echo "========================================="
echo ""

echo "Installing base dependencies..."
python3 -m pip install -r requirements.txt

echo ""
echo "========================================="
echo "  Select PyTorch Version:"
echo "  1. CPU (Default - works on all Macs)"
echo "  2. MPS (Apple Silicon M1/M2/M3 - faster)"
echo "========================================="
echo ""

read -p "Enter 1 or 2: " choice

if [ "$choice" = "2" ]; then
    echo ""
    echo "Installing PyTorch with MPS support..."
    python3 -m pip uninstall torch torchvision -y
    python3 -m pip install torch torchvision
    echo "MPS version installed!"
else
    echo ""
    echo "CPU version already installed!"
fi

echo ""
echo "========================================="
echo "  Installation Complete!"
echo "  Starting server..."
echo "========================================="
echo ""

python3 -m uvicorn app.main:app --reload --port 8000
