import os
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict


def validate_parquet_file(file_path: str) -> Tuple[bool, str]:
    """
    Validate a single parquet file
    Returns: (is_valid, error_message)
    """
    try:
        # Try to read the file
        df = pd.read_parquet(file_path)
        # Check if we can access basic properties
        _ = df.shape
        _ = df.columns
        return True, ""
    except Exception as e:
        return False, str(e)


def find_corrupted_parquet_files(directory: str, pattern: str = "*.parquet") -> Dict[str, List[str]]:
    """
    Find all corrupted parquet files in a directory
    Returns dict with 'valid' and 'corrupted' file lists
    """
    path = Path(directory)
    parquet_files = list(path.rglob(pattern))
    
    valid_files = []
    corrupted_files = []
    
    for file_path in parquet_files:
        is_valid, error = validate_parquet_file(str(file_path))
        if is_valid:
            valid_files.append(str(file_path))
        else:
            corrupted_files.append((str(file_path), error))
            print(f"Corrupted file found: {file_path}")
            print(f"  Error: {error}")
    
    return {
        "valid": valid_files,
        "corrupted": corrupted_files,
        "total": len(parquet_files),
        "valid_count": len(valid_files),
        "corrupted_count": len(corrupted_files)
    }


def safe_read_parquet(file_path: str, default=None):
    """
    Safely read a parquet file, returning default if corrupted
    """
    try:
        return pd.read_parquet(file_path)
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")
        return default


if __name__ == "__main__":
    # Test the validator
    import sys
    if len(sys.argv) > 1:
        directory = sys.argv[1]
        results = find_corrupted_parquet_files(directory)
        print(f"\nValidation Summary:")
        print(f"Total files: {results['total']}")
        print(f"Valid files: {results['valid_count']}")
        print(f"Corrupted files: {results['corrupted_count']}")