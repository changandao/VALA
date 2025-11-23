#!/usr/bin/env python3
"""
PLY File Analyzer Tool

This script analyzes PLY (Polygon File Format) files and provides detailed information
about their structure, properties, and data distribution.

Usage:
    python analyze_ply.py <path_to_ply_file>

Example:
    python analyze_ply.py output/3dgs/scannet_langsplat/scene0000_00/point_cloud/iteration_30000/point_cloud.ply
"""

import sys
import os
import argparse
import numpy as np
from plyfile import PlyData, PlyElement
from collections import Counter
import pandas as pd


def format_bytes(size):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def analyze_property_stats(data, prop_name, prop_type):
    """Analyze statistics for a property"""
    stats = {}
    
    if prop_type in ['f4', 'f8', 'float32', 'float64']:  # Float types
        stats['min'] = float(np.min(data))
        stats['max'] = float(np.max(data))
        stats['mean'] = float(np.mean(data))
        stats['std'] = float(np.std(data))
        stats['median'] = float(np.median(data))
        
        # Check for special values
        nan_count = np.isnan(data).sum()
        inf_count = np.isinf(data).sum()
        if nan_count > 0:
            stats['nan_count'] = int(nan_count)
        if inf_count > 0:
            stats['inf_count'] = int(inf_count)
            
    elif prop_type in ['i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64']:  # Integer types
        stats['min'] = int(np.min(data))
        stats['max'] = int(np.max(data))
        stats['mean'] = float(np.mean(data))
        stats['median'] = float(np.median(data))
        
        # Count unique values (if reasonable number)
        unique_vals = np.unique(data)
        if len(unique_vals) <= 50:
            value_counts = Counter(data)
            stats['unique_values'] = dict(value_counts.most_common())
        else:
            stats['unique_count'] = len(unique_vals)
    
    return stats


def analyze_ply_file(file_path):
    """Main function to analyze PLY file"""
    print(f"Analyzing PLY file: {file_path}")
    print("=" * 80)
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist!")
        return
    
    # Get file size
    file_size = os.path.getsize(file_path)
    print(f"File size: {format_bytes(file_size)}")
    
    try:
        # Read PLY file
        ply_data = PlyData.read(file_path)
        
        print(f"\nPLY file contains {len(ply_data.elements)} element type(s):")
        
        total_data_points = 0
        
        # Analyze each element type
        for element in ply_data.elements:
            element_name = element.name
            element_count = element.count
            total_data_points += element_count
            
            print(f"\n{'─' * 60}")
            print(f"Element: '{element_name}'")
            print(f"Count: {element_count:,} items")
            
            # Get property information
            properties = element.properties
            print(f"Properties: {len(properties)} attributes")
            
            print(f"\nProperty Details:")
            print(f"{'Name':<20} {'Type':<12} {'Shape':<10} {'Description'}")
            print(f"{'-'*20} {'-'*12} {'-'*10} {'-'*20}")
            
            for prop in properties:
                prop_name = prop.name
                prop_val_type = str(prop.val_dtype) if hasattr(prop, 'val_dtype') else str(type(prop))
                
                # Handle list properties
                if hasattr(prop, 'len_dtype'):
                    prop_description = f"List of {prop.val_dtype}"
                    prop_shape = f"List[{prop.val_dtype}]"
                else:
                    prop_description = "Scalar"
                    prop_shape = str(prop_val_type)
                
                print(f"{prop_name:<20} {prop_val_type:<12} {prop_shape:<10} {prop_description}")
            
            # Analyze data statistics for each property
            print(f"\nProperty Statistics:")
            print(f"{'Property':<20} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12}")
            print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
            
            element_data = element.data
            
            for prop in properties:
                prop_name = prop.name
                
                # Skip list properties for now (more complex to analyze)
                if hasattr(prop, 'len_dtype'):
                    print(f"{prop_name:<20} {'<list>':<12} {'<list>':<12} {'<list>':<12} {'<list>'}")
                    continue
                
                try:
                    prop_data = element_data[prop_name]
                    stats = analyze_property_stats(prop_data, prop_name, str(prop.val_dtype))
                    
                    min_val = f"{stats.get('min', 'N/A'):.4f}" if 'min' in stats else "N/A"
                    max_val = f"{stats.get('max', 'N/A'):.4f}" if 'max' in stats else "N/A"
                    mean_val = f"{stats.get('mean', 'N/A'):.4f}" if 'mean' in stats else "N/A"
                    std_val = f"{stats.get('std', 'N/A'):.4f}" if 'std' in stats else "N/A"
                    
                    print(f"{prop_name:<20} {min_val:<12} {max_val:<12} {mean_val:<12} {std_val:<12}")
                    
                    # Print unique values if available
                    if 'unique_values' in stats:
                        print(f"  → Unique values for {prop_name}: {stats['unique_values']}")
                    elif 'unique_count' in stats:
                        print(f"  → Unique value count for {prop_name}: {stats['unique_count']}")
                        
                    # Print special value counts
                    if 'nan_count' in stats:
                        print(f"  → NaN values in {prop_name}: {stats['nan_count']}")
                    if 'inf_count' in stats:
                        print(f"  → Infinite values in {prop_name}: {stats['inf_count']}")
                        
                except Exception as e:
                    print(f"{prop_name:<20} Error: {str(e)}")
        
        print(f"\n{'═' * 80}")
        print(f"SUMMARY:")
        print(f"Total elements: {len(ply_data.elements)}")
        print(f"Total data points: {total_data_points:,}")
        print(f"File size: {format_bytes(file_size)}")
        
        # Calculate average bytes per data point
        if total_data_points > 0:
            bytes_per_point = file_size / total_data_points
            print(f"Average bytes per data point: {bytes_per_point:.2f}")
        
    except Exception as e:
        print(f"Error reading PLY file: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze PLY file properties and structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python analyze_ply.py point_cloud.ply
    python analyze_ply.py output/3dgs/scannet_langsplat/scene0000_00/point_cloud/iteration_30000/point_cloud.ply
        """
    )
    
    parser.add_argument(
        'ply_file',
        help='Path to the PLY file to analyze'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        print("Verbose mode enabled")
    
    analyze_ply_file(args.ply_file)


if __name__ == "__main__":
    main() 