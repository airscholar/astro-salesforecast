# Astro Sales Forecasting MLOps Platform

## Overview

A production-ready MLOps platform for sales forecasting that demonstrates modern machine learning engineering practices. Built on Astronomer (Apache Airflow), this project implements an end-to-end ML pipeline with ensemble modeling, comprehensive visualization, and real-time inference capabilities via Streamlit.

### 🚀 Key Features

- **Automated ML Pipeline**: End-to-end orchestration with Astronomer/Airflow
- **Ensemble Modeling**: Combines XGBoost, LightGBM, and Prophet for robust predictions
- **Advanced Visualizations**: Comprehensive model performance analysis and comparison
- **Real-time Inference**: Streamlit-based web UI for interactive predictions
- **Experiment Tracking**: MLflow integration for model versioning and metrics
- **Distributed Storage**: MinIO S3-compatible object storage for artifacts
- **Containerized Deployment**: Docker-based architecture for consistency

## 🏗️ Architecture

### Technology Stack

| Component | Technology                 | Purpose |
|-----------|----------------------------|---------|  
| **Orchestration** | Astronomer (Airflow 3.0+)  | Workflow automation and scheduling |
| **ML Tracking** | MLflow 2.9+                | Experiment tracking and model registry |
| **Storage** | MinIO                      | S3-compatible artifact storage |
| **ML Models** | XGBoost, LightGBM, Prophet | Ensemble forecasting |
| **Visualization** | Matplotlib, Seaborn, Plotly | Model analysis and insights |
| **Inference UI** | Streamlit                  | Interactive prediction interface |
| **Containerization** | Docker & Docker Compose    | Environment consistency |

## 🚀 Quick Start

### Prerequisites

- Docker Desktop installed and running
- Astronomer CLI (`brew install astro` on macOS, other OS, you can follow the [instructions here](https://www.astronomer.io/docs/astro/cli/install-cli/))
- 8GB+ RAM available for Docker
- Ports 8080, 8501, 5001, 9000, 9001 available

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/airscholar/astro-salesforecast.git
cd Astro-SalesForecast
```

### 2. Start All Services

```bash
# Start Astronomer Airflow services
astro dev start
```

This will start:
- **Airflow UI**: http://localhost:8080 (admin/admin)
- **Streamlit UI**: http://localhost:8501
- **MLflow UI**: http://localhost:5001
- **MinIO Console**: http://localhost:9001 (minioadmin/minioadmin)

### 3. Run the ML Pipeline

1. Open Airflow UI at http://localhost:8080
2. Enable the `sales_forecast_training` DAG
3. Trigger the DAG manually or wait for scheduled run
4. Monitor progress in the Airflow UI

### 4. Use the Inference UI

1. Open Streamlit at http://localhost:8501
2. Click "Load/Reload Models" in the sidebar
3. Choose input method (upload CSV, manual entry, or sample data)
4. Configure forecast parameters
5. Generate predictions and export results

## 📊 ML Pipeline Features

### Data Processing
- Synthetic data generation with realistic patterns
- Time-based train/validation/test splitting
- Comprehensive data validation and quality checks
- Advanced feature engineering (lags, rolling stats, seasonality)

### Model Training
- **XGBoost**: Gradient boosting for non-linear patterns
- **LightGBM**: Fast training with categorical support
- **Ensemble**: Optimized weighted average of all models
- Hyperparameter tuning with Optuna

### Visualization Suite
- Model performance comparison charts
- Time series predictions with confidence intervals
- Residual analysis and diagnostics
- Feature importance rankings
- Interactive plots with Plotly

### Model Management
- Automated experiment tracking with MLflow
- Model versioning and registry
- Artifact storage in MinIO
- Production model promotion workflow

## 🎯 Inference System

### Streamlit Features
- **Multiple Input Methods**: CSV upload, manual entry, sample data
- **Model Selection**: Individual models or ensemble
- **Interactive Visualizations**: Real-time prediction plots
- **Confidence Intervals**: 95% prediction bounds
- **Export Capabilities**: Download predictions as CSV

### API Architecture
```python
# Simplified prediction flow
Input Data → Feature Engineering → Model Prediction → Visualization → Export
```

## 📈 Performance & Metrics

- **Training Time**: ~2-5 minutes for full pipeline
- **Prediction Latency**: <100ms per forecast
- **Model Accuracy**: MAPE < 5% on test data
- **Ensemble Performance**: 15-20% improvement over individual models


## 🐛 Troubleshooting

### Common Issues

1. **Services not starting**: Check Docker memory allocation (8GB minimum)
2. **Models not loading**: Ensure training DAG has completed successfully
3. **Port conflicts**: Stop conflicting services or modify ports in docker-compose
4. **MLflow connection**: Verify MLflow service is running and accessible

### Logs and Debugging

```bash
# Check Airflow logs
astro dev logs

# Check specific service logs
docker-compose -f docker-compose.override.yml logs mlflow
docker-compose -f docker-compose.override.yml logs streamlit
```

## 📚 Documentation

- [Detailed Architecture](docs/ARCHITECTURE.md)
- [Astronomer Docs](https://www.astronomer.io/docs/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.