import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import pandera as pa
from pandera import Column, DataFrameSchema, Check
import yaml
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DataValidator:
    def __init__(self, config_path: str = "/usr/local/airflow/include/config/ml_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.validation_config = self.config['validation']
        self.required_columns = self.validation_config['required_columns']
        self.data_types = self.validation_config['data_types']
        self.value_ranges = self.validation_config['value_ranges']
        
    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        errors = []
        
        # Check required columns
        missing_columns = set(self.required_columns) - set(df.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check data types
        for col, expected_type in self.data_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if actual_type != expected_type:
                    try:
                        if expected_type == "datetime64[ns]":
                            df[col] = pd.to_datetime(df[col])
                        else:
                            df[col] = df[col].astype(expected_type)
                    except Exception as e:
                        errors.append(f"Column {col}: Cannot convert {actual_type} to {expected_type}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        quality_report = {
            "total_rows": len(df),
            "column_stats": {},
            "quality_issues": []
        }
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            quality_report["quality_issues"].append(f"Found {duplicates} duplicate rows")
        
        # Column-wise statistics and checks
        for col in df.columns:
            col_stats = {
                "null_count": df[col].isnull().sum(),
                "null_percentage": (df[col].isnull().sum() / len(df)) * 100,
                "unique_values": df[col].nunique()
            }
            
            if df[col].dtype in ['int64', 'float64']:
                col_stats.update({
                    "mean": df[col].mean(),
                    "std": df[col].std(),
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "outliers": self._detect_outliers(df[col])
                })
                
                # Check value ranges
                if col in self.value_ranges:
                    range_config = self.value_ranges[col]
                    if 'min' in range_config and df[col].min() < range_config['min']:
                        quality_report["quality_issues"].append(
                            f"{col}: Values below minimum ({df[col].min()} < {range_config['min']})"
                        )
                    if 'max' in range_config and df[col].max() > range_config['max']:
                        quality_report["quality_issues"].append(
                            f"{col}: Values above maximum ({df[col].max()} > {range_config['max']})"
                        )
            
            quality_report["column_stats"][col] = col_stats
        
        return quality_report
    
    def _detect_outliers(self, series: pd.Series, method: str = "iqr") -> int:
        if method == "iqr":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return ((series < lower_bound) | (series > upper_bound)).sum()
        elif method == "zscore":
            z_scores = np.abs((series - series.mean()) / series.std())
            return (z_scores > 3).sum()
        else:
            return 0
    
    def create_pandera_schema(self) -> DataFrameSchema:
        schema_dict = {}
        
        for col, dtype in self.data_types.items():
            checks = []
            
            # Add value range checks
            if col in self.value_ranges:
                range_config = self.value_ranges[col]
                if 'min' in range_config:
                    checks.append(Check.greater_than_or_equal_to(range_config['min']))
                if 'max' in range_config:
                    checks.append(Check.less_than_or_equal_to(range_config['max']))
            
            # Map data types
            pandera_dtype = dtype
            if dtype == "datetime64[ns]":
                pandera_dtype = "datetime64"
            
            schema_dict[col] = Column(pandera_dtype, checks=checks, nullable=True)
        
        return DataFrameSchema(schema_dict)
    
    def validate_time_series(self, df: pd.DataFrame, date_col: str = 'date',
                           group_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        ts_report = {
            "date_range": {},
            "frequency_issues": [],
            "gaps": []
        }
        
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Overall date range
        ts_report["date_range"] = {
            "start": df[date_col].min().strftime("%Y-%m-%d"),
            "end": df[date_col].max().strftime("%Y-%m-%d"),
            "days": (df[date_col].max() - df[date_col].min()).days
        }
        
        # Check for gaps in time series
        if group_cols:
            for group, group_df in df.groupby(group_cols):
                sorted_dates = group_df[date_col].sort_values()
                date_diffs = sorted_dates.diff()
                
                # Find gaps (more than 1 day difference)
                gaps = date_diffs[date_diffs > pd.Timedelta(days=1)]
                if len(gaps) > 0:
                    ts_report["gaps"].append({
                        "group": group,
                        "gap_count": len(gaps),
                        "max_gap_days": gaps.max().days
                    })
        else:
            sorted_dates = df[date_col].sort_values()
            date_diffs = sorted_dates.diff()
            gaps = date_diffs[date_diffs > pd.Timedelta(days=1)]
            if len(gaps) > 0:
                ts_report["gaps"] = {
                    "gap_count": len(gaps),
                    "max_gap_days": gaps.max().days,
                    "gap_dates": gaps.index.tolist()
                }
        
        return ts_report
    
    def validate_prediction_data(self, df: pd.DataFrame, 
                               training_stats: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        
        # Check if prediction data has same schema
        is_valid, schema_errors = self.validate_schema(df)
        errors.extend(schema_errors)
        
        # Check for distribution shift
        for col in df.select_dtypes(include=[np.number]).columns:
            if col in training_stats:
                train_mean = training_stats[col]['mean']
                train_std = training_stats[col]['std']
                
                pred_mean = df[col].mean()
                pred_std = df[col].std()
                
                # Simple check for significant distribution shift
                if abs(pred_mean - train_mean) > 3 * train_std:
                    errors.append(
                        f"Potential distribution shift in {col}: "
                        f"mean changed from {train_mean:.2f} to {pred_mean:.2f}"
                    )
        
        return len(errors) == 0, errors
    
    def generate_validation_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Starting data validation")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "dataset_info": {
                "rows": len(df),
                "columns": len(df.columns),
                "memory_usage": df.memory_usage(deep=True).sum() / 1024**2  # MB
            }
        }
        
        # Schema validation
        is_valid, schema_errors = self.validate_schema(df)
        report["schema_validation"] = {
            "is_valid": is_valid,
            "errors": schema_errors
        }
        
        # Data quality validation
        report["data_quality"] = self.validate_data_quality(df)
        
        # Time series validation
        if 'date' in df.columns:
            report["time_series_validation"] = self.validate_time_series(df)
        
        logger.info(f"Validation complete. Valid: {is_valid}")
        return report