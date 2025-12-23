#!/usr/bin/env python3
"""
Generate text.txt files in sparse/0/ directories for each scene.
The content of each text.txt file contains the image names (without suffix)
from the corresponding label directory.
"""

import os
from pathlib import Path


def generate_text_files():
    # Base paths
    base_path = Path("/home/sen.wang/projects/VALA/dataset/3dgs/lerf_ovs")
    label_base_path = base_path / "label"
    
    # Scene names
    scenes = ["teatime", "figurines", "ramen", "waldo_kitchen"]
    
    for scene in scenes:
        # Label directory for this scene
        label_dir = label_base_path / scene
        
        # Target sparse/0/ directory
        sparse_dir = base_path / scene / "sparse" / "0"
        
        # Check if label directory exists
        if not label_dir.exists():
            print(f"Warning: Label directory not found for {scene}: {label_dir}")
            continue
        
        # Check if sparse/0/ directory exists
        if not sparse_dir.exists():
            print(f"Warning: Sparse directory not found for {scene}: {sparse_dir}")
            continue
        
        # Get all image files (jpg/png) and extract names without suffix
        image_names = []
        for file in sorted(label_dir.iterdir()):
            if file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                image_names.append(file.stem)  # stem gives filename without suffix
        
        # Write to text.txt
        text_file_path = sparse_dir / "test.txt"
        with open(text_file_path, 'w') as f:
            f.write('\n'.join(image_names))
        
        print(f"Created {text_file_path}")
        print(f"  Content: {image_names}")
        print()


if __name__ == "__main__":
    generate_text_files()
    print("Done!")

