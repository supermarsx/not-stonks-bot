#!/bin/bash

# Download all repository chunks from the workspace
# This script will help you download the complete not-stonks-bot repository

echo "🚀 Not-Stonks-Bot Repository Download Script"
echo "=============================================="
echo ""
echo "This script downloads the complete trading system in manageable chunks."
echo ""

# Create download directory
DOWNLOAD_DIR="not-stonks-bot-download"
mkdir -p $DOWNLOAD_DIR

echo "📁 Created download directory: $DOWNLOAD_DIR"
echo ""

# Download all chunks
CHUNKS=(
    "INDEX.md"
    "chunk_01"
    "chunk_dir_01_analytics-backend"
    "chunk_dir_02_browser"
    "chunk_dir_03_crawlers"
    "chunk_dir_04_docs"
    "chunk_dir_05_external_api"
    "chunk_dir_06_matrix-trading-command-center"
    "chunk_dir_07_performance"
    "chunk_dir_08_research"
    "chunk_dir_09_testing"
    "chunk_dir_10_tests"
    "chunk_dir_11_trading-command-center"
    "chunk_dir_12_trading_orchestrator"
)

echo "📥 Downloading chunks..."
echo ""

for chunk in "${CHUNKS[@]}"; do
    echo "  → Downloading $chunk..."
    cp -r "/workspace/download_chunks/$chunk" "$DOWNLOAD_DIR/" 2>/dev/null || echo "    ⚠️  Could not copy $chunk"
done

echo ""
echo "✅ Download completed!"
echo ""
echo "📂 Your files are in: ./$DOWNLOAD_DIR/"
echo ""
echo "📋 Next steps:"
echo "1. cd $DOWNLOAD_DIR"
echo "2. Read INDEX.md for detailed instructions"
echo "3. Follow the setup guide to get started"
echo ""
echo "🐍 Quick start:"
echo "   cd $DOWNLOAD_DIR"
echo "   pip install -r requirements.txt"
echo "   cp config.example.json config.json"
echo "   python main.py"
echo ""
echo "📖 For complete setup instructions, see INDEX.md"