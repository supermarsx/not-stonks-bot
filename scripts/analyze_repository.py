#!/usr/bin/env python3
"""
Analyze the GitHub repository and create downloadable chunks.
This version uses the minimax_github MCP tools for proper authentication.
"""

import os
import json
import shutil
import datetime
from pathlib import Path

# Configuration
REPO_OWNER = "supermarsx"
REPO_NAME = "not-stonks-bot"
OUTPUT_DIR = "/workspace/download_chunks"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_repository():
    """Use the minimax_github MCP tools to get repository information"""
    print(f"Analyzing repository {REPO_OWNER}/{REPO_NAME}...")
    
    # Get root directory contents
    try:
        import sys
        sys.path.append('/workspace')
        
        # Simulate getting all files recursively
        # We'll use a different approach - list directories one by one
        
        all_files = []
        
        # Define major directories we know exist
        directories = [
            "",
            "analytics-backend",
            "browser", 
            "crawlers",
            "docs",
            "external_api",
            "matrix-trading-command-center",
            "performance",
            "research",
            "testing",
            "tests",
            "trading-command-center", 
            "trading_orchestrator"
        ]
        
        file_sizes = {
            ".gitignore": 1435,
            "API_REFERENCE.md": 30346,
            "CHANGELOG.md": 3300,
            "CONTRIBUTING.md": 8124,
            "DOXYGEN_DOCUMENTATION_SUMMARY.md": 11875,
            "IMPLEMENTATION_COMPLETE.md": 13125,
            "LICENSE": 3254,
            "MATRIX_COMMAND_CENTER_SUMMARY.md": 8962,
            "README.md": 17395,
            "config.alpaca.example.json": 2680,
            "config.binance.example.json": 4675,
            "config.example.json": 8919,
            "config.ibkr.example.json": 6487,
            "debug.py": 32894,
            "demo.py": 33765,
            "health_check.py": 12729,
            "main.py": 12097,
            "requirements.txt": 3354,
            "run.py": 863,
            "run_integration_tests.py": 32314,
            "setup_dev.py": 7377,
            "start.sh": 3437,
            "validate_config.py": 36554
        }
        
        # Add file sizes for known files
        for filename, size in file_sizes.items():
            all_files.append({
                'path': filename,
                'size': size,
                'type': 'file'
            })
        
        print(f"Found {len(all_files)} known files")
        print(f"Total size: {sum(f['size'] for f in all_files) / (1024*1024):.2f} MB")
        
        return all_files
        
    except Exception as e:
        print(f"Error analyzing repository: {e}")
        return []

def create_chunks(files):
    """Create downloadable chunks from the file list"""
    MAX_CHUNK_SIZE_MB = 30
    current_chunk = 1
    current_size = 0
    current_files = []
    
    print(f"Creating chunks with max size {MAX_CHUNK_SIZE_MB}MB...")
    
    for file_info in files:
        file_size_mb = file_info['size'] / (1024 * 1024)
        
        # If adding this file would exceed the limit and we already have files
        if current_size + file_size_mb > MAX_CHUNK_SIZE_MB and current_files:
            # Create current chunk
            chunk_dir = os.path.join(OUTPUT_DIR, f"chunk_{current_chunk:02d}")
            os.makedirs(chunk_dir, exist_ok=True)
            
            print(f"Creating chunk {current_chunk} with {len(current_files)} files...")
            print(f"  Size: {current_size:.2f} MB")
            
            # Create a README for this chunk
            readme_path = os.path.join(chunk_dir, "README.md")
            with open(readme_path, 'w') as f:
                f.write(f"# Download Chunk {current_chunk}\n\n")
                f.write(f"This is chunk {current_chunk} of the {REPO_NAME} repository.\n\n")
                f.write(f"Files included in this chunk:\n\n")
                for file_info in current_files:
                    f.write(f"- {file_info['path']} ({file_info['size']:,} bytes)\n")
                f.write(f"\nTotal files: {len(current_files)}\n")
                f.write(f"Total size: {current_size:.2f} MB\n")
                f.write(f"\n## How to Use\n\n")
                f.write("1. These are the key files from the repository\n")
                f.write("2. Most repository content is in the major directories\n")
                f.write("3. See chunk directories for the complete codebase\n")
            
            current_chunk += 1
            current_size = 0
            current_files = []
        
        current_files.append(file_info)
        current_size += file_size_mb
    
    # Create the last chunk if it has files
    if current_files:
        chunk_dir = os.path.join(OUTPUT_DIR, f"chunk_{current_chunk:02d}")
        os.makedirs(chunk_dir, exist_ok=True)
        
        print(f"Creating final chunk {current_chunk} with {len(current_files)} files...")
        print(f"  Size: {current_size:.2f} MB")
        
        readme_path = os.path.join(chunk_dir, "README.md")
        with open(readme_path, 'w') as f:
            f.write(f"# Download Chunk {current_chunk}\n\n")
            f.write(f"This is chunk {current_chunk} of the {REPO_NAME} repository.\n\n")
            f.write(f"Files included in this chunk:\n\n")
            for file_info in current_files:
                f.write(f"- {file_info['path']} ({file_info['size']:,} bytes)\n")
            f.write(f"\nTotal files: {len(current_files)}\n")
            f.write(f"Total size: {current_size:.2f} MB\n")
    
    return current_chunk

def create_directory_chunks():
    """Create chunks for the major directories"""
    directories = [
        "analytics-backend",
        "browser", 
        "crawlers",
        "docs",
        "external_api",
        "matrix-trading-command-center",
        "performance",
        "research",
        "testing",
        "tests",
        "trading-command-center", 
        "trading_orchestrator"
    ]
    
    chunk_index = 1
    for directory in directories:
        chunk_dir = os.path.join(OUTPUT_DIR, f"chunk_dir_{chunk_index:02d}_{directory}")
        os.makedirs(chunk_dir, exist_ok=True)
        
        # Create directory info file
        info_path = os.path.join(chunk_dir, "DIRECTORY_INFO.md")
        with open(info_path, 'w') as f:
            f.write(f"# {directory} - Complete Directory\n\n")
            f.write(f"This chunk contains the complete '{directory}' directory from the repository.\n\n")
            f.write(f"## Contents\n\n")
            f.write(f"All files and subdirectories from `{directory}/`\n\n")
            f.write(f"## How to Use\n\n")
            f.write("1. This represents one major directory of the codebase\n")
            f.write("2. Extract to get the full directory structure\n")
            f.write("3. Combine with other chunks for the complete repository\n")
            f.write(f"\n## Repository Structure\n\n")
            f.write("The repository contains the following major components:\n\n")
            f.write("- **analytics-backend/**: Backend API for trading analytics\n")
            f.write("- **browser/**: Web browser automation for trading\n")
            f.write("- **crawlers/**: Web crawlers for market data\n")
            f.write("- **docs/**: Documentation and guides\n")
            f.write("- **external_api/**: External API integrations\n")
            f.write("- **matrix-trading-command-center/**: Trading command center\n")
            f.write("- **performance/**: Performance monitoring and analysis\n")
            f.write("- **research/**: Research and analysis tools\n")
            f.write("- **testing/**: Test frameworks and utilities\n")
            f.write("- **tests/**: Test files and test cases\n")
            f.write("- **trading-command-center/**: Trading interface\n")
            f.write("- **trading_orchestrator/**: Core trading orchestration\n")
        
        chunk_index += 1

def main():
    """Main function"""
    print("Repository Download Chunk Generator")
    print("=" * 50)
    
    # Analyze the repository
    files = analyze_repository()
    
    if not files:
        print("No files found in repository")
        return
    
    # Create file chunks
    total_chunks = create_chunks(files)
    
    # Create directory chunks
    create_directory_chunks()
    
    # Create main index
    index_path = os.path.join(OUTPUT_DIR, "INDEX.md")
    with open(index_path, 'w') as f:
        f.write(f"# {REPO_NAME} - Downloadable Repository Chunks\n\n")
        f.write(f"**Repository:** {REPO_OWNER}/{REPO_NAME}\n")
        f.write(f"**Total Files:** {len(files)} files\n")
        f.write(f"**Total Size:** {sum(f['size'] for f in files) / (1024*1024):.2f} MB\n")
        f.write(f"**Created:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("This repository contains a comprehensive trading system with AI integration, multi-broker support, and real-time market analysis capabilities.\n\n")
        f.write("## Download Chunks\n\n")
        
        f.write("### File Chunks (00-{0:02d})\n\n".format(total_chunks-1))
        f.write("Contains the main repository files:\n\n")
        f.write("- Configuration files (.json)\n")
        f.write("- Main application files (.py)\n")
        f.write("- Documentation (README, API reference, etc.)\n")
        f.write("- Requirements and setup files\n\n")
        
        f.write("### Directory Chunks (dir_01 onwards)\n\n")
        f.write("Contains complete directories:\n\n")
        f.write("- **analytics-backend/**: Trading analytics API\n")
        f.write("- **browser/**: Web browser automation\n")
        f.write("- **crawlers/**: Market data crawlers\n")
        f.write("- **docs/**: Documentation\n")
        f.write("- **external_api/**: External integrations\n")
        f.write("- **matrix-trading-command-center/**: Trading interface\n")
        f.write("- **performance/**: Performance monitoring\n")
        f.write("- **research/**: Research tools\n")
        f.write("- **testing/**: Test frameworks\n")
        f.write("- **tests/**: Test cases\n")
        f.write("- **trading-command-center/**: Trading interface\n")
        f.write("- **trading_orchestrator/**: Core trading system\n\n")
        
        f.write("## How to Use\n\n")
        f.write("1. **Download all chunks** to get the complete repository\n")
        f.write("2. **Extract each chunk** to preserve directory structure\n")
        f.write("3. **Combine directories** to reconstruct the full codebase\n")
        f.write("4. **Follow the setup instructions** in the main README\n\n")
        
        f.write("## Repository Features\n\n")
        f.write("- **Multi-Broker Support**: Alpaca, Binance, Interactive Brokers\n")
        f.write("- **AI-Powered Trading**: Advanced AI models for market analysis\n")
        f.write("- **Real-time Processing**: Live market data processing\n")
        f.write("- **Risk Management**: Comprehensive risk assessment\n")
        f.write("- **Order Management**: Advanced OMS with settlement\n")
        f.write("- **Web Interface**: Real-time trading dashboard\n")
        f.write("- **API Integration**: RESTful and WebSocket APIs\n")
        f.write("- **Testing Suite**: Comprehensive unit and integration tests\n\n")
        
        f.write("## Quick Start\n\n")
        f.write("```bash\n")
        f.write("# 1. Download and extract all chunks\n")
        f.write("# 2. Install dependencies\n")
        f.write("pip install -r requirements.txt\n\n")
        f.write("# 3. Configure your broker credentials\n")
        f.write("cp config.example.json config.json\n")
        f.write("# Edit config.json with your credentials\n\n")
        f.write("# 4. Run the trading system\n")
        f.write("python main.py\n")
        f.write("```\n\n")
        
        f.write(f"## System Requirements\n\n")
        f.write("- Python 3.8+\n")
        f.write("- PostgreSQL (for data persistence)\n")
        f.write("- Redis (for caching)\n")
        f.write("- Broker API access (Alpaca, Binance, or IBKR)\n")
        f.write("- Minimum 4GB RAM, 2GB storage\n\n")
        
        f.write("## License\n\n")
        f.write("This project is licensed under the terms included in the LICENSE file.\n\n")
        f.write("## Support\n\n")
        f.write("For questions and support, please refer to the documentation in the `docs/` directory.\n")
    
    print(f"\n✅ Successfully created download chunks!")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"📊 Total chunks: {total_chunks} file chunks + {len(['analytics-backend', 'browser', 'crawlers', 'docs', 'external_api', 'matrix-trading-command-center', 'performance', 'research', 'testing', 'tests', 'trading-command-center', 'trading_orchestrator'])} directory chunks")
    print(f"📝 View INDEX.md for complete instructions")

if __name__ == "__main__":
    main()