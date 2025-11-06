#!/usr/bin/env python3
"""
Package the entire workspace comprehensively into 30MB chunks.
Includes all Python files, documentation, and important assets.
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

def get_all_important_files():
    """Get all important files from workspace"""
    important_extensions = {
        '.py', '.js', '.ts', '.html', '.css', '.json', '.yaml', '.yml',
        '.md', '.txt', '.cfg', '.ini', '.sh', '.bat', '.sql', '.ipynb',
        '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.pdf'
    }
    
    important_files = []
    excluded_patterns = [
        '__pycache__', '*.pyc', '.git', '.DS_Store', '*.log',
        '.env', 'node_modules', '.pytest_cache', 'venv', '.venv',
        'dist', 'build', '.git', '.svn', '__pycache__', '*.pyc',
        '*.log', '.DS_Store', '.idea', '.vscode', '.agent',
        'node_modules', 'workspace', 'browser_use', '.venv',
        'browser/user_data*', 'browser/sessions', 'mcp_downloaded',
        'debug', 'log', 'pyproject.toml', 'external_api', 'pyarmor_runtime_000000',
        'uv.lock', '.pdf_temp', 'pdf_temp'
    ]
    
    for root, dirs, files in os.walk(WORKSPACE_DIR):
        # Filter out excluded directories
        dirs[:] = [d for d in dirs if not any(pattern in d for pattern in excluded_patterns)]
        
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, WORKSPACE_DIR)
            
            # Skip if file is in excluded patterns
            if any(pattern in file_path for pattern in excluded_patterns):
                continue
                
            # Skip if file extension is not important
            _, ext = os.path.splitext(file)
            if ext.lower() not in important_extensions and ext.lower() != '':
                continue
                
            # Skip very large files (>50MB)
            try:
                if os.path.getsize(file_path) > 50 * 1024 * 1024:
                    print(f"   ⚠️  Skipping large file: {rel_path}")
                    continue
            except OSError:
                continue
                
            important_files.append(file_path)
    
    return important_files

def create_multiple_chunks(files, chunk_size_mb):
    """Create multiple chunks of files"""
    chunk_files = []
    current_size = 0
    current_files = []
    chunk_number = 1
    
    for file_path in files:
        try:
            file_size = os.path.getsize(file_path)
            file_size_mb = file_size / (1024 * 1024)
            
            # If adding this file would exceed the limit
            if current_size + file_size_mb > chunk_size_mb and current_files:
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
    
    return chunk_files

def create_chunk_archive(files, output_path):
    """Create a chunk archive from a list of files"""
    try:
        with tarfile.open(output_path, 'w:gz') as tar:
            for file_path in files:
                try:
                    # Create archive name
                    rel_path = os.path.relpath(file_path, WORKSPACE_DIR)
                    tar.add(file_path, arcname=rel_path)
                except (OSError, IOError) as e:
                    print(f"   ⚠️  Could not add {rel_path}: {e}")
                    continue
        return True
    except Exception as e:
        print(f"   ❌ Error creating {output_path}: {e}")
        return False

def package_workspace_comprehensive():
    """Package the workspace comprehensively"""
    print("🚀 Comprehensive Workspace Packaging")
    print("=" * 50)
    print(f"Workspace: {WORKSPACE_DIR}")
    print(f"Chunk Size: {CHUNK_SIZE_MB}MB")
    print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Get all important files
    print("🔍 Scanning workspace for important files...")
    important_files = get_all_important_files()
    
    if not important_files:
        print("❌ No important files found")
        return
    
    print(f"📊 Found {len(important_files)} important files")
    
    # Calculate total size
    total_size = 0
    for file_path in important_files:
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
    chunk_files = create_multiple_chunks(important_files, CHUNK_SIZE_MB)
    
    if not chunk_files:
        print("❌ Failed to create any chunks")
        return
    
    print()
    print("📋 Chunk Summary:")
    
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
    print(f"🗂️  Files per chunk: ~{len(important_files) // len(chunk_files)}")
    
    # Create comprehensive download instructions
    create_comprehensive_instructions(chunk_files, important_files, total_chunk_size_mb)

def create_comprehensive_instructions(chunk_files, all_files, total_size_mb):
    """Create comprehensive download instructions"""
    instructions_path = os.path.join(WORKSPACE_DIR, "DOWNLOAD_INSTRUCTIONS.md")
    
    # Analyze file types
    file_types = {}
    for file_path in all_files:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '':
            ext = 'no extension'
        file_types[ext] = file_types.get(ext, 0) + 1
    
    with open(instructions_path, 'w') as f:
        f.write(f"# 📦 Not-Stonks-Bot Complete Workspace Download\n\n")
        f.write(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Size:** {total_size_mb:.2f} MB\n")
        f.write(f"**Total Chunks:** {len(chunk_files)}\n")
        f.write(f"**Total Files:** {len(all_files)}\n\n")
        
        f.write("## 📥 Download Chunks\n\n")
        f.write("The complete workspace has been packaged into these download chunks:\n\n")
        
        for i, chunk_file in enumerate(chunk_files, 1):
            file_size = os.path.getsize(os.path.join(WORKSPACE_DIR, chunk_file))
            file_size_mb = file_size / (1024 * 1024)
            f.write(f"{i}. **{chunk_file}** ({file_size_mb:.2f} MB)\n")
        
        f.write(f"\n## 🛠️ How to Download & Extract\n\n")
        f.write("### Step 1: Download All Chunks\n")
        f.write("Download all the chunk files from this workspace directory.\n\n")
        
        f.write("### Step 2: Combine Chunks (if multiple)\n")
        f.write("```bash\n")
        f.write("# For multiple chunks, combine them first:\n")
        f.write("cat not-stonks-bot-chunk-*.tar.gz > not-stonks-bot-complete.tar.gz\n")
        f.write("```\n\n")
        
        f.write("### Step 3: Extract\n")
        f.write("```bash\n")
        f.write("# Extract the combined archive\n")
        f.write("tar -xzf not-stonks-bot-complete.tar.gz\n\n")
        f.write("# Or extract individual chunks\n")
        f.write("for chunk in not-stonks-bot-chunk-*.tar.gz; do\n")
        f.write("  tar -xzf \"$chunk\"\n")
        f.write("done\n")
        f.write("```\n\n")
        
        f.write("## 📊 File Breakdown\n\n")
        f.write("**File Types Included:**\n\n")
        for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(all_files)) * 100
            f.write(f"- **{ext}**: {count} files ({percentage:.1f}%)\n")
        
        f.write(f"\n## 🏗️ What's Included\n\n")
        f.write("This comprehensive package includes:\n\n")
        f.write("### Core System\n")
        f.write("- **Trading Engine**: Multi-broker trading orchestration\n")
        f.write("- **AI Models**: Machine learning and deep learning trading algorithms\n")
        f.write("- **Order Management**: Advanced OMS with settlement and tracking\n")
        f.write("- **Risk Management**: Position sizing, stop-loss, and risk assessment\n")
        f.write("- **Performance Analytics**: Real-time P&L and performance metrics\n\n")
        
        f.write("### Broker Integrations\n")
        f.write("- **Alpaca**: Stock and crypto trading\n")
        f.write("- **Binance**: Cryptocurrency trading\n")
        f.write("- **Interactive Brokers**: Global market access\n\n")
        
        f.write("### User Interfaces\n")
        f.write("- **Web Dashboard**: Real-time trading interface\n")
        f.write("- **Terminal Interface**: Command-line trading tools\n")
        f.write("- **Matrix Theme**: Advanced terminal interface\n")
        f.write("- **API Documentation**: Complete API reference\n\n")
        
        f.write("### Data & Analytics\n")
        f.write("- **Market Data**: Real-time and historical data processing\n")
        f.write("- **Web Crawlers**: Automated market data collection\n")
        f.write("- **Analytics Backend**: FastAPI trading analytics\n")
        f.write("- **Performance Monitoring**: System health and performance tracking\n\n")
        
        f.write("### Development & Testing\n")
        f.write("- **Test Suite**: Comprehensive unit and integration tests\n")
        f.write("- **Development Tools**: Debug, validation, and setup utilities\n")
        f.write("- **Documentation**: Complete project documentation\n")
        f.write("- **Configuration**: Example configs for all brokers\n\n")
        
        f.write("## 🚀 Quick Start Guide\n\n")
        f.write("### 1. Install Dependencies\n")
        f.write("```bash\n")
        f.write("cd not-stonks-bot\n")
        f.write("pip install -r requirements.txt\n")
        f.write("```\n\n")
        
        f.write("### 2. Configure System\n")
        f.write("```bash\n")
        f.write("# Copy and edit configuration\n")
        f.write("cp config.example.json config.json\n")
        f.write("# Edit config.json with your broker API keys\n")
        f.write("```\n\n")
        
        f.write("### 3. Run the System\n")
        f.write("```bash\n")
        f.write("# Start the main trading system\n")
        f.write("python main.py\n\n")
        f.write("# Or run specific components\n")
        f.write("python demo.py          # Run demo\n")
        f.write("python health_check.py  # System health\n")
        f.write("python validate_config.py # Validate config\n")
        f.write("```\n\n")
        
        f.write("### 4. Test Installation\n")
        f.write("```bash\n")
        f.write("python -m pytest tests/        # Run tests\n")
        f.write("python run_integration_tests.py # Integration tests\n")
        f.write("```\n\n")
        
        f.write("## 📋 Directory Structure\n\n")
        f.write("```\n")
        f.write("not-stonks-bot/\n")
        f.write("├── 📁 analytics-backend/     # FastAPI trading analytics\n")
        f.write("├── 📁 browser/               # Selenium web automation\n")
        f.write("├── 📁 crawlers/              # Market data crawlers\n")
        f.write("├── 📁 docs/                  # Documentation\n")
        f.write("├── 📁 external_api/          # External API integrations\n")
        f.write("├── 📁 matrix-trading-command-center/  # Trading interface\n")
        f.write("├── 📁 performance/           # Performance monitoring\n")
        f.write("├── 📁 research/              # Research tools\n")
        f.write("├── 📁 testing/               # Test frameworks\n")
        f.write("├── 📁 tests/                 # Test cases\n")
        f.write("├── 📁 trading-command-center/ # Trading interface\n")
        f.write("├── 📁 trading_orchestrator/  # Core trading system\n")
        f.write("│   ├── 📁 ui/                # User interfaces\n")
        f.write("│   ├── 📁 api/               # API layer\n")
        f.write("│   ├── 📁 models/            # Database models\n")
        f.write("│   ├── 📁 utils/             # Utilities\n")
        f.write("│   └── 📁 config/            # Configuration\n")
        f.write("├── 🔧 main.py                # Main application\n")
        f.write("├── 🔧 requirements.txt       # Dependencies\n")
        f.write("├── 📄 config.example.json    # Configuration template\n")
        f.write("├── 📖 README.md              # Project overview\n")
        f.write("└── 📖 API_REFERENCE.md       # API documentation\n")
        f.write("```\n\n")
        
        f.write("## ⚙️ System Requirements\n\n")
        f.write("### Minimum Requirements\n")
        f.write("- **Python**: 3.8 or higher\n")
        f.write("- **RAM**: 4GB minimum, 8GB recommended\n")
        f.write("- **Storage**: 2GB free space\n")
        f.write("- **OS**: Linux, macOS, or Windows 10+\n\n")
        
        f.write("### Required Services\n")
        f.write("- **PostgreSQL**: 12+ (for data persistence)\n")
        f.write("- **Redis**: 6+ (for caching and queues)\n")
        f.write("- **Chrome/Firefox**: For web automation\n")
        f.write("- **Broker APIs**: Alpaca, Binance, or Interactive Brokers\n\n")
        
        f.write("## ⚠️ Important Security Notes\n\n")
        f.write("1. **API Keys**: Never commit real API keys to version control\n")
        f.write("2. **Paper Trading**: Always start with paper trading\n")
        f.write("3. **Risk Management**: Understand the risks before live trading\n")
        f.write("4. **Network Security**: Use secure connections for all API calls\n")
        f.write("5. **Regular Updates**: Keep dependencies updated\n\n")
        
        f.write("## 📚 Learning Resources\n\n")
        f.write("- **README.md**: Start with the main project README\n")
        f.write("- **API_REFERENCE.md**: Complete API documentation\n")
        f.write("- **docs/**: Detailed documentation and guides\n")
        f.write("- **CHANGELOG.md**: Version history and new features\n")
        f.write("- **CONTRIBUTING.md**: Development guidelines\n\n")
        
        f.write("## 🆘 Troubleshooting\n\n")
        f.write("### Common Issues\n")
        f.write("**Import Errors:**\n")
        f.write("```bash\n")
        f.write("pip install -r requirements.txt --upgrade\n")
        f.write("```\n\n")
        
        f.write("**Database Connection:**\n")
        f.write("```bash\n")
        f.write("# Start PostgreSQL\n")
        f.write("sudo systemctl start postgresql\n")
        f.write("# Create database\n")
        f.write("createdb trading_db\n")
        f.write("```\n\n")
        
        f.write("**Configuration Issues:**\n")
        f.write("```bash\n")
        f.write("python validate_config.py\n")
        f.write("```\n\n")
        
        f.write("## 📜 License & Disclaimer\n\n")
        f.write("This project is licensed under the MIT License.\n\n")
        f.write("**DISCLAIMER**: This software is for educational and research purposes. ")
        f.write("Past performance does not guarantee future results. Trading involves risk ")
        f.write("and you should never trade with money you cannot afford to lose. ")
        f.write("The authors are not responsible for any financial losses.\n\n")
        
        f.write("## 💬 Support\n\n")
        f.write("For questions and support:\n")
        f.write("- Check the documentation in the `docs/` directory\n")
        f.write("- Review the test files to understand functionality\n")
        f.write("- Use the `health_check.py` script to diagnose issues\n")
    
    print(f"📋 Created comprehensive download instructions: {instructions_path}")

if __name__ == "__main__":
    package_workspace_comprehensive()