#!/bin/bash
# NetPulse Setup and Run Script for Linux/Unix
# This script sets up the environment and runs NetPulse

set -e

echo "=========================================="
echo "  NetPulse - Network Anomaly Detector"
echo "  Setup and Run Script (Linux/Unix)"
echo "=========================================="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo ""
echo "[1/5] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "  ✓ Python $PYTHON_VERSION found"

echo ""
echo "[2/5] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  ✓ Virtual environment created"
else
    echo "  ✓ Virtual environment already exists"
fi

echo ""
echo "[3/5] Activating virtual environment and installing dependencies..."
source venv/bin/activate
pip install --upgrade pip > /dev/null 2>&1
pip install -e . > /dev/null 2>&1
echo "  ✓ Dependencies installed"

echo ""
echo "[4/5] Creating necessary directories..."
mkdir -p ~/.netpulse/models/current
mkdir -p ~/.netpulse/data
mkdir -p ~/.netpulse/reports
echo "  ✓ Directories created"

echo ""
echo "[5/5] NetPulse is ready!"
echo ""
echo "=========================================="
echo "  Usage Instructions:"
echo "=========================================="
echo ""
echo "  Start monitoring (requires sudo):"
echo "    sudo ./scripts/run.sh start"
echo ""
echo "  Start monitoring on specific interface:"
echo "    sudo ./scripts/run.sh start --interface eth0"
echo ""
echo "  View live status dashboard:"
echo "    ./scripts/run.sh status"
echo ""
echo "  Generate reports:"
echo "    ./scripts/run.sh report --last 24h"
echo "    ./scripts/run.sh report --last 7d --export alerts.csv"
echo ""
echo "  Train models manually:"
echo "    ./scripts/run.sh train"
echo ""
echo "  Run evaluation:"
echo "    ./scripts/run.sh eval"
echo ""
echo "  Edit configuration:"
echo "    ./scripts/run.sh config"
echo ""
echo "  Reset baseline:"
echo "    ./scripts/run.sh reset-baseline"
echo ""
echo "=========================================="

# If a command was passed as argument, execute it
if [ $# -gt 0 ]; then
    echo ""
    echo "Executing: netpulse $@"
    echo ""
    python -m netpulse "$@"
else
    echo "No command specified. Showing help..."
    echo ""
    python -m netpulse --help
fi
