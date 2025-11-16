# =============================================================================
# INSTRUCTIONS.md - Comprehensive Usage Guide for anomdetec.py
# Author: Elliot McGeachie
# License: MIT License
# Version: 1.1.0
# Last Updated: November 16, 2025
# =============================================================================

## Overview

`anomdetec.py` is a robust, universal predictive maintenance anomaly detection script designed for industrial and scientific applications. It employs a hybrid ensemble approach combining:
- Isolation Forest: An unsupervised tree-based outlier detection algorithm.
- Autoencoder: A neural network-based model that detects anomalies via reconstruction errors.

This tool is engineered for scalability, handling large datasets through chunked processing, and includes advanced feature engineering such as lagged variables, rolling correlations, and FFT-based frequency features. It supports dynamic schema inference for data validation and ensures reproducibility through seeded random operations.

Key Outputs:
- A detailed PDF report including performance metrics (F1-score, precision, recall), anomaly ratio, mean anomaly score, SHAP feature importance summary, and dataset information.
- Console output of key metrics for quick assessment.
- Log file (`anomdetec.log`) for detailed runtime information and debugging.

The script is optimized for various hardware configurations (laptops, workstations, servers) via configurable parameters in `config.yaml`. It maintains full functionality across environments while prioritizing efficiency and accuracy.

## Requirements

### Python Version
- Python 3.8 or higher (tested up to 3.12.3).

### Dependencies
Install the required packages using pip:

```bash
pip install numpy pandas scikit-learn matplotlib shap torch pandera pyyaml tqdm fpdf
```
## Installation

Clone or download the repository containing anomdetec.py, config.yaml, and this instructions.md.
Create a virtual environment (recommended for dependency isolation):
```python -m venv venv```
Activate the virtual environment:

On Unix/macOS:
```source venv/bin/activate```

On Windows:
```venv\Scripts\activate```

Install dependencies as above.
Ensure input CSV files are prepared (see Data Preparation below).

## Usage
- Run the script via the command line:
python anomdetec.py --train <train.csv> --test <test.csv> --config <config.yaml> --output <report.pdf>

## Command Line Arguments
--train (required): Path to the training CSV file (normal data for model fitting).
--test (required): Path to the testing CSV file (data to evaluate for anomalies).
--config (optional): Path to configuration YAML file. Default: config.yaml.
--output (optional): Path to output PDF report. Default: pm_report.pdf.

## Configuration
Customize parameters in config.yaml (see the file for details). Key tunable settings include:

Anomaly contamination fraction.
Autoencoder training epochs and architecture.
Feature engineering options (lags, FFT points, rolling windows).
Isolation Forest hyperparameters for grid search.
SHAP sampling for explainability.

If config.yaml is missing, the script falls back to sensible defaults.

## Data Preparation

Format: CSV files with numeric columns (e.g., sensor readings). Non-numeric columns are ignored.
Optional Column: Include a 'cycle_time' column for cycle-based anomaly simulation in ground truth generation (useful for time-series data like machinery cycles).
Handling Issues: The script automatically detects delimiters, replaces infinities/NaNs, and validates for finite values.
Size: Suitable for large files; processed in chunks to avoid memory overflow.
Feature Engineering: Automatically generates lags, correlations, and FFT features from numeric columns to enhance detection.

## Workflow

Data Loading: Chunked CSV reading with schema inference and validation.
Feature Generation: Adds lagged, interaction (correlation), and frequency (FFT) features.
Scaling: Standardizes data for model input.
Model Training:
Grid-searched Isolation Forest on training data.
Autoencoder trained via MSE loss on training data.

Prediction: Ensemble scoring on test data (IF decision + AE reconstruction error).
Evaluation: Simulates ground truth labels (cycle-based if available, random otherwise) and computes metrics.
Explainability: SHAP values for feature importance.
Reporting: Generates PDF with metrics, reasoning, and SHAP plot; logs details.

## Output Interpretation

PDF Report:
Header: Test file info, overall decision (Anomalous/Normal), metrics.
Reasoning: Explanation of detection logic.
SHAP Summary Plot: Beeswarm plot showing feature impacts on anomalies.

Console: Prints F1, precision, recall for immediate feedback.
Log File: Timestamped events, warnings, and errors for auditing.

## Troubleshooting

Memory Errors: Reduce chunk_size in config.
Convergence Issues: Increase autoencoder_epochs or adjust hidden_ratio.
Invalid Data: Ensure CSVs have numeric columns; check log for schema validation errors.
No GPU Detected: Script falls back to CPU; install CUDA for acceleration.
Dependencies Missing: Re-run pip install command.
Custom Delimiters: Script auto-detects, but ensure consistent formatting.

## Best Practices

Tuning: Start with defaults, then iterate on config based on dataset size and domain knowledge.
Validation: Use known anomalous data for testing to verify metrics.
Scalability: For very large datasets, run on cloud instances with GPU.
Integration: Easily integrable into pipelines (e.g., Airflow) via CLI.
Security: Avoid sensitive data in logs; configure logging level if needed.

## Limitations and Future Enhancements

Current: Relies on simulated ground truth for metrics; real labels can be integrated if provided.
Future: Support for custom models, real-time streaming, and multi-modal data (e.g., images).

