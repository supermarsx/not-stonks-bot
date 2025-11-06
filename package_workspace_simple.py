#!/usr/bin/env python3
"""
Simple and reliable workspace packaging script.
Creates 30MB chunks from all files in workspace.
"""

import os
import tarfile
import shutil
from pathlib import Path
import datetime

# Configuration
WORKSPACE_DIR = "/workspace"
CHUNK_SIZE_MB = 30
CHUNK_EXTENSION = ".tar.gz"

def get_all_files_simple():
    """Get all files from workspace (simple approach)"""
    all_files = []
    
    excluded_dirs = {
        '.git', '__pycache__', '.pytest_cache', 'node_modules', 
        'venv', '.venv', 'dist', 'build', '.backups', 
        '.browser_screenshots', '.memory', '.pdf_temp',
        'download_chunks', 'not-stonks-bot-chunk-', 
        '.env', 'pyproject.toml', 'uv.lock'
    }
    
    excluded_files = {
        '.DS_Store', '*.pyc', '*.log', '.gitignore', 
        'pyarmor_runtime_000000', 'debug.log'
    }
    
    for root, dirs, files in os.walk(WORKSPACE_DIR):
        # Remove excluded directories from dirs to prevent walking into them
        dirs[:] = [d for d in dirs if d not in excluded_dirs and not d.startswith('.')]
        
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, WORKSPACE_DIR)
            
            # Skip excluded files
            if any(pattern in file for pattern in excluded_files):
                continue
            
            # Skip very large files (>20MB)
            try:
                if os.path.getsize(file_path) > 20 * 1024 * 1024:
                    print(f"   ⚠️  Skipping large file: {rel_path} ({os.path.getsize(file_path) // (1024*1024)}MB)")
                    continue
            except OSError:
                continue
                
            all_files.append(file_path)
    
    return all_files

def create_chunk_archive(files, output_path):
    """Create a chunk archive from a list of files"""
    try:
        with tarfile.open(output_path, 'w:gz') as tar:
            for file_path in files:
                try:
                    rel_path = os.path.relpath(file_path, WORKSPACE_DIR)
                    tar.add(file_path, arcname=rel_path)
                except (OSError, IOError) as e:
                    print(f"   ⚠️  Could not add {rel_path}: {e}")
                    continue
        return True
    except Exception as e:
        print(f"   ❌ Error creating {output_path}: {e}")
        return False

def package_workspace_simple():
    """Package workspace with simple approach"""
    print("🚀 Simple Workspace Packaging")
    print("=" * 50)
    print(f"Workspace: {WORKSPACE_DIR}")
    print(f"Chunk Size: {CHUNK_SIZE_MB}MB")
    print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Get all files
    print("🔍 Scanning workspace for files...")
    all_files = get_all_files_simple()
    
    if not all_files:
        print("❌ No files found")
        return
    
    print(f"📊 Found {len(all_files)} files")
    
    # Calculate total size
    total_size = 0
    for file_path in all_files:
        try:
            total_size += os.path.getsize(file_path)
        except OSError:
            continue
    
    total_size_mb = total_size / (1024 * 1024)
    print(f"📦 Total size: {total_size_mb:.2f} MB")
    print()
    
    # Remove existing chunks
    existing_chunks = [f for f in os.listdir(WORKSPACE_DIR) if f.startswith('not-stonks-bot-chunk-') and f.endswith(CHUNK_EXTENSION)]
    for chunk in existing_chunks:
        os.remove(os.path.join(WORKSPACE_DIR, chunk))
        print(f"🗑️  Removed old chunk: {chunk}")
    
    if existing_chunks:
        print()
    
    # Create chunks
    print("📦 Creating download chunks...")
    chunk_files = []
    current_size = 0
    current_files = []
    chunk_number = 1
    
    for file_path in all_files:
        try:
            file_size = os.path.getsize(file_path)
            file_size_mb = file_size / (1024 * 1024)
            
            # If adding this file would exceed the limit
            if current_size + file_size_mb > CHUNK_SIZE_MB and current_files:
                # Create current chunk
                chunk_name = f"not-stonks-bot-chunk-{chunk_number:02d}{CHUNK_EXTENSION}"
                chunk_path = os.path.join(WORKSPACE_DIR, chunk_name)
                
                success = create_chunk_archive(current_files, chunk_path)
                if success:
                    actual_size = os.path.getsize(chunk_path) / (1024 * 1024)
                    chunk_files.append(chunk_name)
                    print(f"   ✅ Created: {chunk_name} ({actual_size:.2f} MB, {len(current_files)} files)")
                
                # Reset for next chunk
                current_files = []
                current_size = 0
                chunk_number += 1
            
            current_files.append(file_path)
            current_size += file_size_mb
            
        except OSError:
            print(f"   ⚠️  Could not access file: {file_path}")
            continue
    
    # Create final chunk if it has files
    if current_files:
        chunk_name = f"not-stonks-bot-chunk-{chunk_number:02d}{CHUNK_EXTENSION}"
        chunk_path = os.path.join(WORKSPACE_DIR, chunk_name)
        
        success = create_chunk_archive(current_files, chunk_path)
        if success:
            actual_size = os.path.getsize(chunk_path) / (1024 * 1024)
            chunk_files.append(chunk_name)
            print(f"   ✅ Created: {chunk_name} ({actual_size:.2f} MB, {len(current_files)} files)")
    
    if not chunk_files:
        print("❌ Failed to create any chunks")
        return
    
    print()
    print("📋 Final Chunk Summary:")
    
    total_chunk_size = 0
    for chunk_file in chunk_files:
        chunk_path = os.path.join(WORKSPACE_DIR, chunk_file)
        if os.path.exists(chunk_path):
            file_size = os.path.getsize(chunk_path)
            file_size_mb = file_size / (1024 * 1024)
            total_chunk_size += file_size
            print(f"   📄 {chunk_file}: {file_size_mb:.2f} MB")
    
    total_chunk_size_mb = total_chunk_size / (1024 * 1024)
    print()
    print(f"📊 Total packaged size: {total_chunk_size_mb:.2f} MB")
    print(f"📁 Number of chunks: {len(chunk_files)}")
    print(f"🗂️  Total files: {len(all_files)}")
    
    # Show file type breakdown
    file_types = {}
    for file_path in all_files:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '':
            ext = 'no extension'
        file_types[ext] = file_types.get(ext, 0) + 1
    
    print(f"\n📈 File Type Breakdown:")
    for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True)[:10]:
        percentage = (count / len(all_files)) * 100
        print(f"   {ext}: {count} files ({percentage:.1f}%)")
    
    # Create final instructions
    create_final_instructions(chunk_files, all_files, total_chunk_size_mb)

def create_final_instructions(chunk_files, all_files, total_size_mb):
    """Create final download instructions"""
    instructions_path = os.path.join(WORKSPACE_DIR, "DOWNLOAD_INSTRUCTIONS.md")
    
    with open(instructions_path, 'w') as f:
        f.write(f"# 📦 Not-Stonks-Bot Complete Workspace Download\n\n")
        f.write(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Size:** {total_size_mb:.2f} MB\n")
        f.write(f"**Total Chunks:** {len(chunk_files)}\n")
        f.write(f"**Total Files:** {len(all_files)}\n\n")
        
        f.write("## 📥 Download These Files\n\n")
        f.write("Download ALL of these chunk files from the workspace root:\n\n")
        
        for i, chunk_file in enumerate(chunk_files, 1):
            file_size = os.path.getsize(os.path.join(WORKSPACE_DIR, chunk_file))
            file_size_mb = file_size / (1024 * 1024)
            f.write(f"**{i}. {chunk_file}** ({file_size_mb:.2f} MB)\n")
        
        f.write(f"\n## 🛠️ How to Extract\n\n")
        f.write("### Step 1: Download All Chunks\n")
        f.write("Download all the chunk files from the workspace root directory.\n\n")
        
        f.write("### Step 2: Combine (if multiple chunks)\n")
        f.write("```bash\n")
        f.write("# For multiple chunks, combine them first:\n")
        f.write("cat not-stonks-bot-chunk-*.tar.gz > not-stonks-bot-complete.tar.gz\n")
        f.write("```\n\n")
        
        f.write("### Step 3: Extract\n")
        f.write("```bash\n")
        f.write("# Extract the combined archive or individual chunks\n")
        f.write("tar -xzf not-stonks-bot-complete.tar.gz\n\n")
        f.write("# Or extract each chunk separately:\n")
        f.write("for chunk in not-stonks-bot-chunk-*.tar.gz; do\n")
        f.write("  tar -xzf \"$chunk\"\n")
        f.write("done\n")
        f.write("```\n\n")
        
        f.write("## 🚀 Quick Start\n\n")
        f.write("```bash\n")
        f.write("# After extraction\n")
        f.write("cd not-stonks-bot\n")
        f.write("pip install -r requirements.txt\n")
        f.write("cp config.example.json config.json\n")
        f.write("python main.py\n")
        f.write("```\n\n")
        
        f.write("## 📁 What's Included\n\n")
        f.write("- ✅ **Complete Trading System**: Multi-broker trading platform\n")
        f.write("- ✅ **AI-Powered Analytics**: Machine learning trading algorithms\n")
        f.write("- ✅ **Web Dashboard**: Real-time trading interface\n")
        f.write("- ✅ **APIs**: RESTful and WebSocket APIs\n")
        f.write("- ✅ **Test Suite**: Comprehensive testing framework\n")
        f.write("- ✅ **Documentation**: Complete project documentation\n")
        f.write("- ✅ **Configuration**: All broker configurations\n\n")
        
        f.write("## ⚠️ Important Notes\n\n")
        f.write("- **Educational Purpose**: This is for learning and research\n")
        f.write("- **Start with Paper Trading**: Never start with real money\n")
        f.write("- **Risk Disclaimer**: Trading involves significant risk\n")
        f.write("- **API Keys**: Configure your own broker API keys in config.json\n\n")
        
        f.write("## 📊 Chunk Contents\n\n")
        f.write(f"The chunks contain the complete not-stonks-bot trading system with {len(all_files)} files.\n")
        f.write("Each chunk is a self-contained archive that can be extracted independently.\n")
        f.write("For the complete system, extract all chunks to get all files.\n")
    
    print(f"📋 Created download instructions: {instructions_path}")
    print()
    print("✅ Workspace packaging complete!")
    print()
    print("🎯 Ready to download:")
    for chunk_file in chunk_files:
        file_size = os.path.getsize(os.path.join(WORKSPACE_DIR, chunk_file)) / (1024 * 1024)
        print(f"   📄 {chunk_file} ({file_size:.2f} MB)")

if __name__ == "__main__":
    package_workspace_simple()