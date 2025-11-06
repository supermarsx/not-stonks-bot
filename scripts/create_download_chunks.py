#!/usr/bin/env python3
"""
Create downloadable chunks of 30MB each from the GitHub repository.
This will help users download the large repository in manageable pieces.
"""

import os
import json
import requests
import tarfile
import io
import datetime
from pathlib import Path
import shutil

# Configuration
REPO_OWNER = "supermarsx"
REPO_NAME = "not-stonks-bot"
BRANCH = "main"
MAX_CHUNK_SIZE_MB = 30
OUTPUT_DIR = "/workspace/download_chunks"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_github_file_size(file_path):
    """Get the file size from GitHub API"""
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{file_path}"
    response = requests.get(url, headers={
        'Authorization': f'token {os.environ["GITHUB_TOKEN"]}',
        'Accept': 'application/vnd.github.v3+json'
    })
    if response.status_code == 200:
        return response.json().get('size', 0)
    return 0

def get_all_files_recursively(path=""):
    """Get all files recursively from GitHub repository"""
    files = []
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{path}"
    response = requests.get(url, headers={
        'Authorization': f'token {os.environ["GITHUB_TOKEN"]}',
        'Accept': 'application/vnd.github.v3+json'
    })
    
    if response.status_code != 200:
        print(f"Error fetching {path}: {response.status_code}")
        return files
        
    for item in response.json():
        if item['type'] == 'file':
            file_path = item['path']
            file_size = item['size']
            files.append({
                'path': file_path,
                'size': file_size,
                'download_url': item['download_url']
            })
        elif item['type'] == 'dir':
            files.extend(get_all_files_recursively(item['path']))
    
    return files

def download_file(download_url, local_path):
    """Download a file from GitHub"""
    response = requests.get(download_url)
    if response.status_code == 200:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, 'wb') as f:
            f.write(response.content)
        return True
    return False

def create_chunks():
    """Create downloadable chunks"""
    print(f"Getting all files from {REPO_OWNER}/{REPO_NAME}...")
    all_files = get_all_files_recursively()
    
    print(f"Found {len(all_files)} files in repository")
    
    # Group files by size
    current_chunk = 1
    current_size = 0
    current_files = []
    
    for file_info in all_files:
        file_size_mb = file_info['size'] / (1024 * 1024)
        
        # If adding this file would exceed the limit and we already have files
        if current_size + file_size_mb > MAX_CHUNK_SIZE_MB and current_files:
            # Create current chunk
            chunk_dir = os.path.join(OUTPUT_DIR, f"chunk_{current_chunk:02d}")
            os.makedirs(chunk_dir, exist_ok=True)
            
            print(f"Creating chunk {current_chunk} with {len(current_files)} files...")
            
            # Download all files in this chunk
            for file_info in current_files:
                local_path = os.path.join(chunk_dir, file_info['path'])
                if download_file(file_info['download_url'], local_path):
                    print(f"  Downloaded: {file_info['path']} ({file_info['size']:,} bytes)")
                else:
                    print(f"  Failed to download: {file_info['path']}")
            
            # Create a readme for this chunk
            readme_path = os.path.join(chunk_dir, "README.md")
            with open(readme_path, 'w') as f:
                f.write(f"# Download Chunk {current_chunk}\n\n")
                f.write(f"This is chunk {current_chunk} of the {REPO_NAME} repository.\n\n")
                f.write(f"Files included in this chunk:\n\n")
                for file_info in current_files:
                    f.write(f"- {file_info['path']} ({file_info['size']:,} bytes)\n")
                f.write(f"\nTotal files: {len(current_files)}\n")
                f.write(f"Total size: {current_size:.2f} MB\n")
            
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
        
        for file_info in current_files:
            local_path = os.path.join(chunk_dir, file_info['path'])
            if download_file(file_info['download_url'], local_path):
                print(f"  Downloaded: {file_info['path']} ({file_info['size']:,} bytes)")
            else:
                print(f"  Failed to download: {file_info['path']}")
        
        readme_path = os.path.join(chunk_dir, "README.md")
        with open(readme_path, 'w') as f:
            f.write(f"# Download Chunk {current_chunk}\n\n")
            f.write(f"This is chunk {current_chunk} of the {REPO_NAME} repository.\n\n")
            f.write(f"Files included in this chunk:\n\n")
            for file_info in current_files:
                f.write(f"- {file_info['path']} ({file_info['size']:,} bytes)\n")
            f.write(f"\nTotal files: {len(current_files)}\n")
            f.write(f"Total size: {current_size:.2f} MB\n")
    
    # Create an index file
    index_path = os.path.join(OUTPUT_DIR, "INDEX.md")
    with open(index_path, 'w') as f:
        f.write(f"# {REPO_NAME} - Downloadable Chunks\n\n")
        f.write(f"Repository: {REPO_OWNER}/{REPO_NAME}\n")
        f.write(f"Branch: {BRANCH}\n")
        f.write(f"Created: {datetime.datetime.now().isoformat()}\n\n")
        f.write(f"The repository has been split into {current_chunk} chunks of maximum {MAX_CHUNK_SIZE_MB}MB each.\n\n")
        f.write("## Chunks:\n\n")
        for i in range(1, current_chunk + 1):
            chunk_dir = f"chunk_{i:02d}"
            f.write(f"- [Chunk {i:02d}]({chunk_dir}/) - Part {i} of the repository\n")
        f.write("\n## How to Use\n\n")
        f.write("1. Download the chunk(s) you need\n")
        f.write("2. Extract the contents\n")
        f.write("3. The files are in their original directory structure\n")
        f.write("4. Combine all chunks to get the complete repository\n")
        f.write("\n## Installation\n\n")
        f.write("```bash\n")
        f.write("# Download and extract all chunks\n")
        f.write("for i in {01.." + str(current_chunk) + "}; do\n")
        f.write("  tar -xzf chunk_$i.tar.gz\n")
        f.write("done\n")
        f.write("```\n")
    
    print(f"\nCreated {current_chunk} chunks in {OUTPUT_DIR}")
    print(f"Total repository size: {sum(f['size'] for f in all_files) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    create_chunks()