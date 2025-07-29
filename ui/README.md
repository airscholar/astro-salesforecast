# Sales Forecasting UI

A Streamlit-based web interface for sales forecasting using trained ML models.

## Features

- 🔮 **Model Inference**: Generate sales forecasts using trained models
- 📊 **Multiple Input Methods**: Upload CSV, manual entry, or use sample data
- 📈 **Interactive Visualizations**: View predictions with confidence intervals
- 🤖 **Model Selection**: Choose between ensemble, XGBoost, or LightGBM
- 💾 **Export Results**: Download predictions as CSV
- 🎯 **Real-time Predictions**: Get instant forecasts with loaded models

## Quick Start

### Using Docker Compose (Recommended)

The UI is included in the main docker-compose setup:

```bash
# Start all services including UI
docker-compose -f docker-compose.override.yml up -d

# Access the UI
open http://localhost:8501
```

### Local Development

```bash
# Navigate to UI directory
cd ui

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export MLFLOW_TRACKING_URI=http://localhost:5001
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin

# Run the app
streamlit run inference_app.py
```

## Usage

1. **Load Models**: Click "Load/Reload Models" in the sidebar
2. **Select Input Method**:
   - Upload CSV with historical sales data
   - Enter recent sales manually
   - Generate sample data for testing
3. **Configure Forecast**:
   - Choose model type (ensemble recommended)
   - Set forecast horizon (1-90 days)
4. **Generate Predictions**: Click "Run Prediction"
5. **Export Results**: Download forecast as CSV

## Input Data Format

CSV files should contain:
- `date`: Date column (YYYY-MM-DD format)
- `sales`: Sales amount (numeric)
- `store_id`: Store identifier (optional)

Example:
```csv
date,store_id,sales
2024-01-01,store_001,5234.50
2024-01-02,store_001,4892.75
```

## Models

The UI supports three model types:
- **Ensemble**: Combines XGBoost and LightGBM (recommended)
- **XGBoost**: Gradient boosting model
- **LightGBM**: Light gradient boosting model

## Architecture

```
ui/
├── app.py              # Main multi-page app
├── inference_app.py    # Simplified inference-only app
├── pages/
│   └── inference.py    # Inference page for multi-page app
├── utils/
│   └── model_loader.py # Model loading and prediction utilities
├── requirements.txt    # Python dependencies
├── Dockerfile         # Container configuration
└── README.md         # This file
```

## Environment Variables

- `MLFLOW_TRACKING_URI`: MLflow server URL (default: http://mlflow:5001)
- `MLFLOW_S3_ENDPOINT_URL`: MinIO endpoint (default: http://minio:9000)
- `AWS_ACCESS_KEY_ID`: MinIO access key
- `AWS_SECRET_ACCESS_KEY`: MinIO secret key

## Troubleshooting

### Models not loading
- Ensure MLflow service is running
- Check that models have been trained (run training DAG first)
- Verify network connectivity between services

### Predictions failing
- Check input data format
- Ensure all required columns are present
- Verify model compatibility with input features

### UI not accessible
- Check if port 8501 is available
- Verify Docker container is running: `docker ps`
- Check logs: `docker logs sales-forecasting_420ea7-streamlit-ui-1`