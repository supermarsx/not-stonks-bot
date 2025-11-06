#!/usr/bin/env python3
"""
Package the entire workspace into 30MB chunks for easy download.
Creates compressed archives in the workspace root.
"""

import os
import tarfile
import gzip
import shutil
from pathlib import Path
import datetime

# Configuration
WORKSPACE_DIR = "/workspace"
CHUNK_SIZE_MB = 30
CHUNK_EXTENSION = ".tar.gz"

def get_directory_size(path):
    """Get the total size of a directory in bytes"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(filepath)
            except (OSError, IOError):
                continue
    return total_size

def create_tar_archive(source_dir, output_path, exclude_patterns=None):
    """Create a tar.gz archive of a directory"""
    if exclude_patterns is None:
        exclude_patterns = [
            '__pycache__',
            '*.pyc',
            '.git',
            '.DS_Store',
            'node_modules',
            '.pytest_cache',
            '*.log',
            '.env',
            'download_chunks',
            '*.tar.gz',
            'not-stonks-bot-download-*'
        ]
    
    try:
        with tarfile.open(output_path, 'w:gz') as tar:
            # Add files to the archive
            for root, dirs, files in os.walk(source_dir):
                # Skip excluded directories
                dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]
                
                for file in files:
                    file_path = os.path.join(root, file)
                    # Skip excluded files
                    if any(pattern in file for pattern in exclude_patterns):
                        continue
                    
                    try:
                        # Add file to archive
                        tar.add(file_path, arcname=os.path.relpath(file_path, source_dir))
                    except (OSError, IOError):
                        continue
        return True
    except Exception as e:
        print(f"Error creating archive {output_path}: {e}")
        return False

def package_workspace():
    """Package the workspace into 30MB chunks"""
    print("🚀 Workspace Packaging System")
    print("=" * 50)
    print(f"Workspace: {WORKSPACE_DIR}")
    print(f"Chunk Size: {CHUNK_SIZE_MB}MB")
    print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check workspace size
    workspace_size_mb = get_directory_size(WORKSPACE_DIR) / (1024 * 1024)
    print(f"📊 Total workspace size: {workspace_size_mb:.2f} MB")
    
    # Remove any existing chunks
    chunk_files = [f for f in os.listdir(WORKSPACE_DIR) if f.startswith('not-stonks-bot-chunk-') and f.endswith(CHUNK_EXTENSION)]
    for chunk_file in chunk_files:
        chunk_path = os.path.join(WORKSPACE_DIR, chunk_file)
        os.remove(chunk_path)
        print(f"🗑️  Removed old chunk: {chunk_file}")
    
    if chunk_files:
        print()
    
    # Create a single comprehensive archive first
    chunk_name = f"not-stonks-bot-complete{CHUNK_EXTENSION}"
    chunk_path = os.path.join(WORKSPACE_DIR, chunk_name)
    
    print(f"📦 Creating complete archive: {chunk_name}")
    print("   This may take a few minutes...")
    
    # Check if it's within size limit
    if workspace_size_mb <= CHUNK_SIZE_MB:
        print("   ✅ Workspace fits in single chunk")
        success = create_tar_archive(WORKSPACE_DIR, chunk_path)
        if success:
            file_size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
            print(f"   ✅ Created: {chunk_name} ({file_size_mb:.2f} MB)")
        else:
            print("   ❌ Failed to create archive")
            return
    else:
        print(f"   ⚠️  Workspace is {workspace_size_mb:.2f}MB, exceeds {CHUNK_SIZE_MB}MB limit")
        print("   📝 Will be split into multiple chunks")
        
        # For large workspaces, we'll need to split
        # Let's try to create the archive and then split it
        temp_name = f"not-stonks-bot-temp{CHUNK_EXTENSION}"
        temp_path = os.path.join(WORKSPACE_DIR, temp_name)
        
        success = create_tar_archive(WORKSPACE_DIR, temp_path)
        if success:
            # Split the large archive into 30MB chunks
            chunk_number = 1
            chunk_bytes = CHUNK_SIZE_MB * 1024 * 1024
            
            with open(temp_path, 'rb') as input_file:
                while True:
                    chunk_filename = f"not-stonks-bot-chunk-{chunk_number:02d}{CHUNK_EXTENSION}"
                    chunk_filepath = os.path.join(WORKSPACE_DIR, chunk_filename)
                    
                    with open(chunk_filepath, 'wb') as output_file:
                        bytes_written = 0
                        while bytes_written < chunk_bytes:
                            chunk = input_file.read(min(chunk_bytes - bytes_written, 1024 * 1024))
                            if not chunk:
                                break
                            output_file.write(chunk)
                            bytes_written += len(chunk)
                    
                    file_size_mb = os.path.getsize(chunk_filepath) / (1024 * 1024)
                    print(f"   ✅ Created: {chunk_filename} ({file_size_mb:.2f} MB)")
                    
                    if bytes_written < chunk_bytes:
                        break
                    
                    chunk_number += 1
                    
                    if chunk_number > 50:  # Safety limit
                        print("   ⚠️  Safety limit reached (50 chunks)")
                        break
            
            # Remove temporary file
            os.remove(temp_path)
            print(f"   🗑️  Cleaned up temporary file")
        else:
            print("   ❌ Failed to create temporary archive")
            return
    
    print()
    print("📋 Chunk Information:")
    
    # List all created chunks
    chunk_files = [f for f in os.listdir(WORKSPACE_DIR) if f.startswith('not-stonks-bot-chunk-') and f.endswith(CHUNK_EXTENSION)]
    chunk_files.extend([f for f in os.listdir(WORKSPACE_DIR) if f.startswith('not-stonks-bot-complete') and f.endswith(CHUNK_EXTENSION)])
    chunk_files.sort()
    
    total_size = 0
    for chunk_file in chunk_files:
        chunk_path = os.path.join(WORKSPACE_DIR, chunk_file)
        if os.path.exists(chunk_path):
            file_size = os.path.getsize(chunk_path)
            file_size_mb = file_size / (1024 * 1024)
            total_size += file_size
            print(f"   📄 {chunk_file}: {file_size_mb:.2f} MB")
    
    total_size_mb = total_size / (1024 * 1024)
    print()
    print(f"📊 Total packaged size: {total_size_mb:.2f} MB")
    print(f"📁 Number of chunks: {len(chunk_files)}")
    
    # Create download instructions
    create_download_instructions(chunk_files, total_size_mb)

def create_download_instructions(chunk_files, total_size_mb):
    """Create download instructions file"""
    instructions_path = os.path.join(WORKSPACE_DIR, "DOWNLOAD_INSTRUCTIONS.md")
    
    with open(instructions_path, 'w') as f:
        f.write(f"# 📦 Not-Stonks-Bot Workspace Download\n\n")
        f.write(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Size:** {total_size_mb:.2f} MB\n")
        f.write(f"**Total Chunks:** {len(chunk_files)}\n\n")
        
        f.write("## 📥 How to Download\n\n")
        f.write("The workspace has been packaged into downloadable chunks below:\n\n")
        
        for i, chunk_file in enumerate(chunk_files, 1):
            file_size = os.path.getsize(os.path.join(WORKSPACE_DIR, chunk_file))
            file_size_mb = file_size / (1024 * 1024)
            f.write(f"{i}. **{chunk_file}** ({file_size_mb:.2f} MB)\n")
        
        f.write(f"\n## 🛠️ How to Extract\n\n")
        f.write("### Option 1: Complete Archive (if single file)\n")
        f.write("```bash\n")
        f.write("tar -xzf not-stonks-bot-complete.tar.gz\n")
        f.write("```\n\n")
        
        f.write("### Option 2: Multiple Chunks\n")
        f.write("```bash\n")
        f.write("# Download all chunk files\n")
        f.write("# Then combine and extract:\n")
        f.write("cat not-stonks-bot-chunk-* > not-stonks-bot-combined.tar.gz\n")
        f.write("tar -xzf not-stonks-bot-combined.tar.gz\n")
        f.write("rm not-stonks-bot-combined.tar.gz  # Cleanup\n")
        f.write("```\n\n")
        
        f.write("## 📋 What's Included\n\n")
        f.write("This package contains the complete not-stonks-bot trading system:\n\n")
        f.write("- **Trading Engine**: Multi-broker trading system\n")
        f.write("- **AI Models**: Machine learning trading algorithms\n")
        f.write("- **Web Interface**: Trading dashboard and controls\n")
        f.write("- **APIs**: RESTful and WebSocket APIs\n")
        f.write("- **Documentation**: Complete setup and usage guides\n")
        f.write("- **Tests**: Comprehensive test suite\n")
        f.write("- **Configuration**: Example configs for all brokers\n\n")
        
        f.write("## 🚀 Quick Start\n\n")
        f.write("1. **Extract the archive**\n")
        f.write("2. **Install dependencies:** `pip install -r requirements.txt`\n")
        f.write("3. **Configure:** `cp config.example.json config.json`\n")
        f.write("4. **Run:** `python main.py`\n\n")
        
        f.write("## ⚠️ Important Notes\n\n")
        f.write("- **Security**: Configure API keys in config.json\n")
        f.write("- **Testing**: Start with paper trading\n")
        f.write("- **Risk**: This is for educational/research purposes\n")
        f.write("- **Support**: See documentation in extracted files\n\n")
        
        f.write("## 📁 File Structure\n\n")
        f.write("```\n")
        f.write("not-stonks-bot/\n")
        f.write("├── analytics-backend/     # Trading analytics API\n")
        f.write("├── browser/               # Web browser automation\n")
        f.write("├── crawlers/              # Market data crawlers\n")
        f.write("├── docs/                  # Documentation\n")
        f.write("├── external_api/          # External integrations\n")
        f.write("├── matrix-trading-command-center/  # Trading interface\n")
        f.write("├── performance/           # Performance monitoring\n")
        f.write("├── research/              # Research tools\n")
        f.write("├── testing/               # Test frameworks\n")
        f.write("├── tests/                 # Test cases\n")
        f.write("├── trading-command-center/ # Trading interface\n")
        f.write("├── trading_orchestrator/  # Core trading system\n")
        f.write("├── main.py                # Main application\n")
        f.write("├── requirements.txt       # Dependencies\n")
        f.write("└── config.example.json    # Configuration example\n")
        f.write("```\n")
    
    print(f"📋 Created download instructions: {instructions_path}")

if __name__ == "__main__":
    package_workspace()