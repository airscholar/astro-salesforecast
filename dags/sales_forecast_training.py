from datetime import datetime, timedelta
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import pandas as pd
import os
import sys

# Add include path
sys.path.append("/usr/local/airflow/include")

from ml_models.train_models import ModelTrainer
from utils.mlflow_utils import MLflowManager
from data_validation.validators import DataValidator


default_args = {
    "owner": "data-team",
    "depends_on_past": False,
    "start_date": datetime(2025, 7, 22),
    "email_on_failure": True,
    "email_on_retry": False,
    "email": ["admin@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def extract_data(**context):
    """Extract sales data from source"""
    from utils.data_generator import RealisticSalesDataGenerator
    
    # Generate realistic sales data
    # In production, this would be replaced with actual data extraction
    data_output_dir = "/tmp/sales_data"
    
    # For training, generate 3 years of historical data for better patterns
    generator = RealisticSalesDataGenerator(
        start_date="2021-01-01",
        end_date="2021-12-31"
    )
    
    print("Generating realistic sales data...")
    file_paths = generator.generate_sales_data(output_dir=data_output_dir)
    
    # Log statistics
    total_files = sum(len(paths) for paths in file_paths.values())
    print(f"Generated {total_files} files:")
    for data_type, paths in file_paths.items():
        print(f"  - {data_type}: {len(paths)} files")
    
    # Push metadata to XCom
    context["task_instance"].xcom_push(key="data_output_dir", value=data_output_dir)
    context["task_instance"].xcom_push(key="file_paths", value=file_paths)
    context["task_instance"].xcom_push(key="total_files", value=total_files)
    
    return f"Data extraction completed - {total_files} files generated"


def validate_data(**context):
    """Validate the extracted data"""
    import glob
    
    data_output_dir = context["task_instance"].xcom_pull(
        task_ids="extract_data", key="data_output_dir"
    )
    file_paths = context["task_instance"].xcom_pull(
        task_ids="extract_data", key="file_paths"
    )
    
    validator = DataValidator()
    validation_reports = []
    total_rows = 0
    issues_found = []
    
    # Validate sales files (main dataset)
    print(f"Validating {len(file_paths['sales'])} sales files...")
    
    for i, sales_file in enumerate(file_paths['sales'][:10]):  # Sample first 10 files
        df = pd.read_parquet(sales_file)
        
        # For sales data, we need to adjust the expected columns
        if i == 0:  # Log first file schema
            print(f"Sales data columns: {df.columns.tolist()}")
        
        # Basic validation
        if df.empty:
            issues_found.append(f"Empty file: {sales_file}")
            continue
            
        # Check for required columns in sales data
        required_cols = ['date', 'store_id', 'product_id', 'quantity_sold', 'revenue']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            issues_found.append(f"Missing columns in {sales_file}: {missing_cols}")
            
        total_rows += len(df)
        
        # Validate data quality
        if df['quantity_sold'].min() < 0:
            issues_found.append(f"Negative quantities in {sales_file}")
        if df['revenue'].min() < 0:
            issues_found.append(f"Negative revenue in {sales_file}")
    
    # Validate supplementary data
    for data_type in ['promotions', 'store_events', 'customer_traffic']:
        if data_type in file_paths and file_paths[data_type]:
            sample_file = file_paths[data_type][0]
            df = pd.read_parquet(sample_file)
            print(f"{data_type} data shape: {df.shape}")
            print(f"{data_type} columns: {df.columns.tolist()}")
    
    # Create summary report
    validation_summary = {
        "total_files_validated": len(file_paths['sales'][:10]),
        "total_rows": total_rows,
        "issues_found": len(issues_found),
        "issues": issues_found[:5]  # First 5 issues
    }
    
    if issues_found:
        print(f"Validation completed with {len(issues_found)} issues:")
        for issue in issues_found[:5]:
            print(f"  - {issue}")
    else:
        print(f"Validation passed! Total rows: {total_rows}")
    
    context["task_instance"].xcom_push(key="validation_summary", value=validation_summary)
    
    return "Data validation completed"


def train_models(**context):
    """Train all models"""
    data_output_dir = context["task_instance"].xcom_pull(
        task_ids="extract_data", key="data_output_dir"
    )
    file_paths = context["task_instance"].xcom_pull(
        task_ids="extract_data", key="file_paths"
    )
    
    # Load and combine sales data
    print("Loading sales data from multiple files...")
    sales_dfs = []
    
    # Load all sales files (or a sample for large datasets)
    max_files = 50  # Limit for training demo
    for i, sales_file in enumerate(file_paths['sales'][:max_files]):
        df = pd.read_parquet(sales_file)
        sales_dfs.append(df)
        if (i + 1) % 10 == 0:
            print(f"  Loaded {i + 1} files...")
    
    # Combine all sales data
    sales_df = pd.concat(sales_dfs, ignore_index=True)
    print(f"Combined sales data shape: {sales_df.shape}")
    
    # Aggregate to daily level per store/product
    # For forecasting, we typically want daily totals
    daily_sales = sales_df.groupby(['date', 'store_id', 'product_id', 'category']).agg({
        'quantity_sold': 'sum',
        'revenue': 'sum',
        'cost': 'sum',
        'profit': 'sum',
        'discount_percent': 'mean',
        'unit_price': 'mean'
    }).reset_index()
    
    # Rename revenue to sales for compatibility with existing trainer
    daily_sales = daily_sales.rename(columns={'revenue': 'sales'})
    
    # Load supplementary data
    if file_paths.get('promotions'):
        promo_df = pd.read_parquet(file_paths['promotions'][0])
        # Create promotion indicator
        promo_summary = promo_df.groupby(['date', 'product_id'])['discount_percent'].max().reset_index()
        promo_summary['has_promotion'] = 1
        daily_sales = daily_sales.merge(
            promo_summary[['date', 'product_id', 'has_promotion']], 
            on=['date', 'product_id'], 
            how='left'
        )
        daily_sales['has_promotion'] = daily_sales['has_promotion'].fillna(0)
    
    if file_paths.get('customer_traffic'):
        # Load and aggregate traffic data
        traffic_dfs = []
        for traffic_file in file_paths['customer_traffic'][:10]:
            traffic_dfs.append(pd.read_parquet(traffic_file))
        traffic_df = pd.concat(traffic_dfs, ignore_index=True)
        
        traffic_summary = traffic_df.groupby(['date', 'store_id']).agg({
            'customer_traffic': 'sum',
            'is_holiday': 'max'
        }).reset_index()
        
        daily_sales = daily_sales.merge(
            traffic_summary,
            on=['date', 'store_id'],
            how='left'
        )
    
    print(f"Final training data shape: {daily_sales.shape}")
    print(f"Columns: {daily_sales.columns.tolist()}")
    
    # Initialize trainer with MLflow
    trainer = ModelTrainer()
    
    # For the trainer, we'll focus on aggregate store-level sales
    # This is more typical for business forecasting
    store_daily_sales = daily_sales.groupby(['date', 'store_id']).agg({
        'sales': 'sum',
        'quantity_sold': 'sum',
        'profit': 'sum',
        'has_promotion': 'mean',  # Proportion of products on promotion
        'customer_traffic': 'first',
        'is_holiday': 'first'
    }).reset_index()
    
    # Ensure date column is datetime
    store_daily_sales['date'] = pd.to_datetime(store_daily_sales['date'])
    
    # Split data
    train_df, val_df, test_df = trainer.prepare_data(
        store_daily_sales,
        target_col="sales",
        date_col="date",
        group_cols=["store_id"],
        categorical_cols=["store_id"]
    )
    
    print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}")
    
    # Train models with MLflow tracking
    results = trainer.train_all_models(
        train_df, val_df, test_df, 
        target_col="sales", 
        use_optuna=True  # Enable hyperparameter tuning for better performance
    )
    
    # Log results
    for model_name, model_results in results.items():
        if "metrics" in model_results:
            print(f"\n{model_name} metrics:")
            for metric, value in model_results["metrics"].items():
                print(f"  {metric}: {value:.4f}")
    
    print("\nVisualization charts have been generated and saved to MLflow/MinIO")
    print("Charts include:")
    print("  - Model metrics comparison")
    print("  - Predictions vs actual values")
    print("  - Residuals analysis")
    print("  - Error distribution")
    print("  - Feature importance comparison")
    
    # Create serializable results (remove model objects)
    serializable_results = {}
    for model_name, model_results in results.items():
        serializable_results[model_name] = {
            'metrics': model_results.get('metrics', {})
        }
        # Don't include the actual model object or predictions as they can't be serialized
    
    # Get the current MLflow run ID
    import mlflow
    current_run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
    
    context["task_instance"].xcom_push(key="training_results", value=serializable_results)
    if current_run_id:
        context["task_instance"].xcom_push(key="mlflow_run_id", value=current_run_id)
    
    return "Model training completed"


def evaluate_models(**context):
    """Evaluate and compare models"""
    results = context["task_instance"].xcom_pull(
        task_ids="train_models", key="training_results"
    )

    mlflow_manager = MLflowManager()

    # Find best model based on RMSE
    best_model_name = None
    best_rmse = float("inf")

    for model_name, model_results in results.items():
        if "metrics" in model_results and "rmse" in model_results["metrics"]:
            if model_results["metrics"]["rmse"] < best_rmse:
                best_rmse = model_results["metrics"]["rmse"]
                best_model_name = model_name

    print(f"Best model: {best_model_name} with RMSE: {best_rmse:.4f}")

    # Get the best run
    best_run = mlflow_manager.get_best_model(metric="rmse", ascending=True)

    context["task_instance"].xcom_push(key="best_model", value=best_model_name)
    context["task_instance"].xcom_push(key="best_run_id", value=best_run["run_id"])

    return f"Best model: {best_model_name}"


def register_best_model(**context):
    """Register the best model in MLflow Model Registry"""
    best_model = context["task_instance"].xcom_pull(
        task_ids="evaluate_models", key="best_model"
    )
    run_id = context["task_instance"].xcom_pull(
        task_ids="evaluate_models", key="best_run_id"
    )

    mlflow_manager = MLflowManager()

    # Register models
    model_versions = {}
    for model_name in ["xgboost", "lightgbm"]:
        version = mlflow_manager.register_model(run_id, model_name, model_name)
        model_versions[model_name] = version
        print(f"Registered {model_name} version: {version}")

    context["task_instance"].xcom_push(key="model_versions", value=model_versions)

    return "Models registered successfully"


def transition_to_production(**context):
    """Transition best model to production"""
    model_versions = context["task_instance"].xcom_pull(
        task_ids="register_model", key="model_versions"
    )

    mlflow_manager = MLflowManager()

    # Transition models to production
    for model_name, version in model_versions.items():
        mlflow_manager.transition_model_stage(model_name, version, "Production")
        print(f"Transitioned {model_name} v{version} to Production")

    return "Models transitioned to production"


def generate_performance_report(**context):
    """Generate model performance report"""
    results = context["task_instance"].xcom_pull(
        task_ids="train_models", key="training_results"
    )
    validation_summary = context["task_instance"].xcom_pull(
        task_ids="validate_data", key="validation_summary"
    )

    report = {
        "timestamp": datetime.now().isoformat(),
        "data_summary": {
            "total_rows": validation_summary.get("total_rows", 0) if validation_summary else 0,
            "files_validated": validation_summary.get("total_files_validated", 0) if validation_summary else 0,
            "issues_found": validation_summary.get("issues_found", 0) if validation_summary else 0,
            "issues": validation_summary.get("issues", []) if validation_summary else []
        },
        "model_performance": {},
    }

    if results:
        for model_name, model_results in results.items():
            if "metrics" in model_results:
                report["model_performance"][model_name] = model_results["metrics"]

    # Save report
    import json

    with open("/tmp/performance_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Performance report generated")
    print(f"Models trained: {list(report['model_performance'].keys())}")
    return report


# Create DAG
dag = DAG(
    "sales_forecast_training",
    default_args=default_args,
    description="Train sales forecasting models",
    schedule="@weekly",  # Run weekly
    catchup=False,
    tags=["ml", "training", "sales"],
)

# Define tasks
extract_task = PythonOperator(
    task_id="extract_data", python_callable=extract_data, dag=dag
)

validate_task = PythonOperator(
    task_id="validate_data", python_callable=validate_data, dag=dag
)

train_task = PythonOperator(
    task_id="train_models", python_callable=train_models, dag=dag
)

evaluate_task = PythonOperator(
    task_id="evaluate_models", python_callable=evaluate_models, dag=dag
)

register_task = PythonOperator(
    task_id="register_model", python_callable=register_best_model, dag=dag
)

transition_task = PythonOperator(
    task_id="transition_to_production",
    python_callable=transition_to_production,
    dag=dag,
)

report_task = PythonOperator(
    task_id="generate_report", python_callable=generate_performance_report, dag=dag
)

# Cleanup task
cleanup_task = BashOperator(
    task_id="cleanup",
    bash_command="rm -rf /tmp/sales_data /tmp/performance_report.json || true",
    dag=dag,
)

# Define dependencies
(
    extract_task
    >> validate_task
    >> train_task
    >> evaluate_task
    >> register_task
    >> transition_task
)
[train_task, validate_task] >> report_task >> cleanup_task
