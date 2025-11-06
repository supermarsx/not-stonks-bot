#!/bin/bash
# Day Trading Orchestrator - Linux/macOS Startup Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Banner
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║           DAY TRADING ORCHESTRATOR STARTUP                   ║"
echo "║                    Linux/macOS Script                        ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

print_status "Starting Day Trading Orchestrator..."
print_status "Script directory: $SCRIPT_DIR"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed or not in PATH"
    exit 1
fi

print_success "Python 3 found: $(python3 --version)"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_warning "Virtual environment not found"
    print_status "Creating virtual environment..."
    
    if python3 -m venv venv; then
        print_success "Virtual environment created"
    else
        print_error "Failed to create virtual environment"
        exit 1
    fi
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
if [ ! -f "requirements_installed" ]; then
    print_status "Installing dependencies..."
    if pip install -r trading_orchestrator/requirements.txt; then
        touch requirements_installed
        print_success "Dependencies installed"
    else
        print_error "Failed to install dependencies"
        exit 1
    fi
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p logs data backups

# Set environment variables
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Start the application
print_status "Launching Day Trading Orchestrator..."
echo ""

# Handle different startup modes
case "${1:-normal}" in
    "demo")
        print_status "Starting in DEMO mode..."
        python3 main.py --demo
        ;;
    "debug")
        print_status "Starting in DEBUG mode..."
        python3 -m pdb main.py
        ;;
    "create-config")
        print_status "Creating default configuration..."
        python3 main.py --create-config
        print_warning "Please edit config.json with your API keys before starting!"
        ;;
    "health-check")
        print_status "Running health check..."
        python3 health_check.py
        ;;
    "normal"|*)
        print_status "Starting in NORMAL mode..."
        python3 main.py
        ;;
esac

print_success "Application shutdown complete"