#!/usr/bin/env python3
"""
Script to check for corrupted Parquet files in the data directory
Usage: python check_parquet_files.py /path/to/data
"""

import sys
import os
from pathlib import Path

# Add include path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'include'))

from utils.parquet_validator import find_corrupted_parquet_files


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_parquet_files.py /path/to/data")
        sys.exit(1)
    
    data_directory = sys.argv[1]
    
    if not os.path.exists(data_directory):
        print(f"Error: Directory {data_directory} does not exist")
        sys.exit(1)
    
    print(f"Checking Parquet files in: {data_directory}")
    print("-" * 50)
    
    # Check for corrupted files
    results = find_corrupted_parquet_files(data_directory)
    
    print(f"\nValidation Summary:")
    print(f"Total files checked: {results['total']}")
    print(f"Valid files: {results['valid_count']}")
    print(f"Corrupted files: {results['corrupted_count']}")
    
    if results['corrupted_count'] > 0:
        print(f"\nCorrupted files found:")
        for file_path, error in results['corrupted']:
            print(f"\n  File: {file_path}")
            print(f"  Error: {error}")
        
        print(f"\nRecommendations:")
        print("1. Re-generate or re-download the corrupted files")
        print("2. Check if the files were truncated during transfer")
        print("3. Verify disk space and permissions")
        print("4. Consider removing corrupted files from the dataset")
    else:
        print("\nAll Parquet files are valid!")


if __name__ == "__main__":
    main()