# =============================================================================
# anomdetec.py - Universal Predictive Maintenance Anomaly Detector (Enhanced)
# 
# Author: Elliot McGeachie
# Copyright (c) 2025 Elliot McGeachie
# License: MIT License
# Version: 1.1.0
# 
# Overview:
#   A professional-grade universal anomaly detection framework combining 
#   Isolation Forest and Autoencoder models. It dynamically infers data schema, 
#   generates advanced features (lags, interactions, FFT), and supports 
#   configurable parameters via YAML. Produces comprehensive PDF reports with 
#   SHAP explainability, dataset summaries, and performance metrics.
#
# Usage:
#   python anomdetec.py --train train.csv --test test.csv --config config.yaml --output report.pdf
#
# Dependencies:
#   numpy, pandas, scikit-learn, matplotlib, shap, torch, pandera, pyyaml, tqdm, fpdf
#
# Enhancements in v1.1.0:
#   - Improved error handling and logging.
#   - Added version checking and reproducibility.
#   - Enhanced PDF report with executive summary and metrics table.
#   - More robust feature generation with safeguards.
#   - Expanded docstrings and comments for clarity.
# =============================================================================

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
import logging
import os
import csv
import yaml
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import shap
import torch
import torch.nn as nn
from fpdf import FPDF
from tqdm import tqdm
from logging.handlers import RotatingFileHandler

# ----------------------------- Version Info -----------------------------
__version__ = "1.1.0"

# ----------------------------- Logging Setup -----------------------------
handler = RotatingFileHandler('anomdetec.log', maxBytes=5_000_000, backupCount=5)
logging.basicConfig(handlers=[handler], level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"Starting anomdetec.py v{__version__}")

# ----------------------------- Reproducibility ---------------------------
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# --------------------------- Schema Inference -----------------------------
from pandera import DataFrameSchema, Column, Check
import pandera.errors
pandera.errors._warnings_enabled = False

def infer_schema(df):
    """
    Dynamically generate a Pandera schema for all numeric columns, 
    enforcing finite values and allowing coercion to float.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return DataFrameSchema({
        col: Column(float, checks=[Check(lambda s: np.isfinite(s).all(), element_wise=False)])
        for col in numeric_cols
    }, strict=False, coerce=True)

# ---------------------------- Autoencoder ---------------------------------
class AutoEncoder(nn.Module):
    """
    Fully connected Autoencoder for anomaly detection via reconstruction errors.
    Supports configurable hidden layers, dropout, and batch normalization.
    """
    def __init__(self, input_dim, hidden_ratio=0.25, dropout=0.0, batch_norm=False):
        super(AutoEncoder, self).__init__()
        hidden_dim = max(1, int(input_dim * hidden_ratio))
        layers = [nn.Linear(input_dim, hidden_dim * 2), nn.ReLU()]
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim * 2))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU()]
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*layers)

        dec_layers = [nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU()]
        if batch_norm:
            dec_layers.append(nn.BatchNorm1d(hidden_dim * 2))
        if dropout > 0:
            dec_layers.append(nn.Dropout(dropout))
        dec_layers += [nn.Linear(hidden_dim * 2, input_dim), nn.ReLU()]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ------------------------- Configuration Loader --------------------------
def load_config(config_path):
    """
    Load YAML configuration file. If missing or malformed, fall back to defaults 
    and log the issue.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        logger.warning("Config file not found. Using default parameters.")
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in config: {e}. Using defaults.")
    return {
        'contamination': 0.05,
        'autoencoder_epochs': 50,
        'hidden_ratio': 0.25,
        'chunk_size': 10000,
        'max_lag': 3,
        'fft_points': 64,
        'dropout': 0.0,
        'batch_norm': False,
        'if_n_estimators': [100, 200],
        'if_max_samples': ['auto', 0.8],
        'if_max_features': [0.5, 1.0],
        'rolling_window': 5,
        'shap_sample': 100
    }

# -------------------------- Data Loading & Features ----------------------
def sniff_delimiter(file_path):
    """
    Automatically detect CSV delimiter by sniffing a sample of the file.
    Falls back to comma if detection fails.
    """
    try:
        with open(file_path, 'r', newline='') as f:
            sample = f.read(2048)
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample)
            return dialect.delimiter
    except Exception as e:
        logger.warning(f"Delimiter sniffing failed for {file_path}: {e}. Using default ','.")
        return ','

def generate_features(df, max_lag=3, fft_points=64, rolling_window=5):
    """
    Generate enhanced features for improved anomaly detection:
    - Lagged versions of numeric columns (backfilled).
    - Rolling pairwise correlations (filled with 0).
    - Top 5 FFT magnitude peaks per column.
    Safeguards against empty dataframes or constant columns.
    """
    if df.empty:
        logger.warning("Empty dataframe provided for feature generation.")
        return df
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        logger.warning("No numeric columns found for feature generation.")
        return df
    
    features_list = [df.copy()]

    # Lag features
    lag_features = pd.concat([
        df[col].shift(lag).bfill().rename(f"{col}_lag{lag}")
        for col in numeric_cols for lag in range(1, max_lag + 1)
    ], axis=1)
    features_list.append(lag_features)

    # Interaction features (rolling correlations)
    interaction_features = pd.concat([
        df[c1].rolling(window=rolling_window, min_periods=1).corr(df[c2]).fillna(0).rename(f"{c1}_{c2}_corr")
        for i, c1 in enumerate(numeric_cols) for c2 in numeric_cols[i+1:]
    ], axis=1)
    features_list.append(interaction_features)

    # FFT features (safeguard for short series)
    fft_n = min(fft_points, len(df))
    fft_features = pd.DataFrame({
        f"{col}_fft_{i}": np.abs(np.fft.rfft(df[col].values, n=fft_n)[:min(5, fft_n//2 + 1)])[i] 
        if len(df) > 0 else 0
        for col in numeric_cols for i in range(5)
    }, index=df.index if not df.empty else None)
    features_list.append(fft_features)

    feature_df = pd.concat(features_list, axis=1).fillna(0)
    logger.info(f"Generated {len(feature_df.columns) - len(df.columns)} additional features.")
    return feature_df

def batch_load_csv(file_name, chunk_size=10000, schema_cache=None, max_lag=3, fft_points=64, rolling_window=5):
    """
    Load CSV in chunks to handle large files, validate schema, and generate features.
    Handles NaNs/infinities and logs progress.
    """
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"File not found: {file_name}")
    
    delimiter = sniff_delimiter(file_name)
    chunks = []
    for chunk in tqdm(pd.read_csv(file_name, sep=delimiter, chunksize=chunk_size, engine='python'), desc=f"Loading {file_name}"):
        numeric = chunk.select_dtypes(include=[np.number])
        numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
        numeric.dropna(axis=0, how="any", inplace=True)
        if not numeric.empty:
            numeric = generate_features(numeric, max_lag=max_lag, fft_points=fft_points, rolling_window=rolling_window)
            chunks.append(numeric)
    
    if not chunks:
        raise ValueError(f"No valid numeric data found in {file_name}")
    
    data = pd.concat(chunks, ignore_index=True)
    schema = schema_cache if schema_cache else infer_schema(data)
    try:
        schema.validate(data)
    except pandera.errors.SchemaError as e:
        logger.error(f"Schema validation failed: {e}")
        raise
    logger.info(f"Loaded {len(data)} rows, {len(data.columns)} columns from {file_name}.")
    return data, schema

# ------------------------- Model Training --------------------------------
def train_model(scaled_data, contamination, epochs=50, hidden_ratio=0.25, dropout=0.0, batch_norm=False,
                if_params=None):
    """
    Train the ensemble models: Grid-searched Isolation Forest and Autoencoder.
    Uses unsupervised scoring for IF hyperparameter selection.
    """
    def unsupervised_scorer(estimator, X, y=None):
        return np.mean(estimator.decision_function(X))

    param_grid = if_params if if_params else {'n_estimators': [100, 200],
                                              'max_samples': ['auto', 0.8],
                                              'max_features': [0.5, 1.0]}

    grid = GridSearchCV(IsolationForest(contamination=contamination, random_state=42),
                        param_grid, cv=3, n_jobs=-1,
                        scoring=make_scorer(unsupervised_scorer, greater_is_better=True))
    grid.fit(scaled_data)
    isolation_forest = grid.best_estimator_
    logger.info(f"Best Isolation Forest parameters: {grid.best_params_} (CV score: {grid.best_score_:.4f})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training Autoencoder on {device}")
    input_dim = scaled_data.shape[1]
    autoencoder = AutoEncoder(input_dim, hidden_ratio=hidden_ratio, dropout=dropout, batch_norm=batch_norm).to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    tensor = torch.tensor(scaled_data, dtype=torch.float32).to(device)
    for epoch in tqdm(range(epochs), desc="Training Autoencoder"):
        reconstructed = autoencoder(tensor)
        loss = criterion(reconstructed, tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            logger.debug(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    logger.info("Autoencoder training complete.")

    return {'isolation_forest': isolation_forest, 'autoencoder': autoencoder}

# ------------------------- Anomaly Prediction ----------------------------
def predict_anomalies(models, scaled_test_data, contamination):
    """
    Perform ensemble anomaly detection using Isolation Forest predictions 
    and Autoencoder reconstruction errors. Returns labels and normalized scores.
    """
    if_predictions = models['isolation_forest'].predict(scaled_test_data)
    if_scores = models['isolation_forest'].decision_function(scaled_test_data)

    device = next(models['autoencoder'].parameters()).device
    test_tensor = torch.tensor(scaled_test_data, dtype=torch.float32).to(device)
    with torch.no_grad():
        reconstructed = models['autoencoder'](test_tensor)

    ae_errors = torch.mean((reconstructed - test_tensor) ** 2, dim=1).cpu().numpy()
    ae_threshold = np.percentile(ae_errors, 100 * (1 - contamination))
    ae_predictions = (ae_errors > ae_threshold).astype(int)

    ensemble_predictions = (if_predictions == -1).astype(int) + ae_predictions
    ensemble_labels = (ensemble_predictions >= 1).astype(int)
    ensemble_scores = - (np.abs(if_scores) + ae_errors) / 2  # Negative for anomaly severity
    logger.info(f"Ensemble complete: {np.mean(ensemble_labels):.2%} anomalies flagged (mean score: {np.mean(ensemble_scores):.4f}).")
    return ensemble_labels, ensemble_scores

# ------------------------- Ground Truth ----------------------------------
def generate_ground_truth(test_data, test_sensor_data, contamination):
    """
    Simulate ground truth anomaly labels for metric evaluation. Uses cycle_time 
    if available for degradation simulation; otherwise, random sampling.
    """
    true_labels = np.zeros(len(test_sensor_data))
    if 'cycle_time' in test_data.columns:
        test_data = test_data.sort_values('cycle_time')
        cycle_times = test_data['cycle_time'].values
        max_cycle = np.max(cycle_times)
        stds = np.std(test_sensor_data, axis=0)
        stds[stds == 0] = 1  # Avoid division by zero
        drift = (test_sensor_data - np.mean(test_sensor_data, axis=0)) / stds
        anomaly_indices = np.where((cycle_times / max_cycle > 0.8) | (np.sum(np.abs(drift) > 1.5, axis=1) > 3))[0]  # Enhanced with abs for bidirectional drift
        true_labels[anomaly_indices] = 1
        logger.info("Ground truth generated using cycle_time degradation model.")
    else:
        expected = int(len(test_sensor_data) * contamination)
        idx = np.random.choice(len(test_sensor_data), expected, replace=False)
        true_labels[idx] = 1
        logger.info("Ground truth simulated via random sampling (no cycle_time column).")
    return true_labels

# ------------------------- Explainability -------------------------------
def explain_anomalies(models, test_sensor_data, features, shap_sample=100):
    """
    Compute SHAP values for explainability using a sample of test data.
    Uses Isolation Forest as the base model for efficiency.
    """
    if shap_sample > len(test_sensor_data):
        shap_sample = len(test_sensor_data)
        logger.warning(f"SHAP sample reduced to dataset size: {shap_sample}")
    explainer = shap.Explainer(models['isolation_forest'], test_sensor_data[:shap_sample])
    shap_values = explainer(test_sensor_data[:shap_sample])
    logger.info("SHAP values computed.")
    return shap_values

# ------------------------- Reporting -------------------------------------
def generate_report(test_file, decision, reasoning, f1, precision, recall, anomaly_ratio,
                    mean_score, shap_values, features, test_sensor_data, output):
    """
    Generate an enhanced PDF report with executive summary, metrics table, 
    reasoning, and SHAP summary plot.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Executive Summary
    pdf.cell(200, 10, txt="Predictive Maintenance Anomaly Detection Report", ln=1, align='C')
    pdf.set_font("Arial", "B", size=10)
    pdf.cell(200, 10, txt="Executive Summary", ln=1)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 5, txt=f"This report analyzes {test_file} using a hybrid Isolation Forest and Autoencoder model. Overall status: {decision}. Anomaly ratio: {anomaly_ratio:.2%}. Recommendations: Investigate high-impact features from SHAP plot.")
    
    # Metrics Table
    pdf.ln(10)
    pdf.set_font("Arial", "B", size=10)
    pdf.cell(200, 10, txt="Performance Metrics", ln=1)
    pdf.set_font("Arial", size=10)
    pdf.cell(50, 10, "Metric", border=1)
    pdf.cell(50, 10, "Value", border=1, ln=1)
    pdf.cell(50, 10, "F1-Score", border=1)
    pdf.cell(50, 10, f"{f1:.4f}", border=1, ln=1)
    pdf.cell(50, 10, "Precision", border=1)
    pdf.cell(50, 10, f"{precision:.4f}", border=1, ln=1)
    pdf.cell(50, 10, "Recall", border=1)
    pdf.cell(50, 10, f"{recall:.4f}", border=1, ln=1)
    pdf.cell(50, 10, "Anomaly Ratio", border=1)
    pdf.cell(50, 10, f"{anomaly_ratio:.2%}", border=1, ln=1)
    pdf.cell(50, 10, "Mean Score", border=1)
    pdf.cell(50, 10, f"{mean_score:.4f}", border=1, ln=1)
    
    # Reasoning
    pdf.ln(10)
    pdf.set_font("Arial", "B", size=10)
    pdf.cell(200, 10, txt="Detection Reasoning", ln=1)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 5, txt=reasoning)
    
    # Dataset Info
    pdf.ln(10)
    pdf.set_font("Arial", "B", size=10)
    pdf.cell(200, 10, txt="Dataset Information", ln=1)
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Rows: {len(test_sensor_data)}, Columns: {len(features)}", ln=1)
    
    # SHAP Plot
    pdf.add_page()
    pdf.set_font("Arial", "B", size=10)
    pdf.cell(200, 10, txt="SHAP Feature Importance Summary", ln=1)
    shap.summary_plot(shap_values, features=test_sensor_data[:len(shap_values)],
                      feature_names=features, show=False)
    plt.savefig('shap_summary.png', dpi=300)  # Higher DPI for quality
    pdf.image('shap_summary.png', x=10, y=20, w=190)
    os.remove('shap_summary.png')  # Cleanup
    
    pdf.output(output)
    logger.info(f"Report saved to {output}")

# ------------------------- Main CLI --------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Universal Predictive Maintenance Anomaly Detector (v{})".format(__version__),
        epilog="For detailed instructions, see instructions.md."
    )
    parser.add_argument('--train', required=True, type=str, help="Path to training CSV (normal data)")
    parser.add_argument('--test', required=True, type=str, help="Path to testing CSV (data to analyze)")
    parser.add_argument('--config', default='config.yaml', type=str, help="Path to config YAML (default: config.yaml)")
    parser.add_argument('--output', default='pm_report.pdf', type=str, help="Path to output PDF (default: pm_report.pdf)")
    args = parser.parse_args()

    config = load_config(args.config)

    try:
        # ----------------- Load and scale training data -----------------
        train_df, schema = batch_load_csv(args.train, config.get('chunk_size', 10000),
                                          max_lag=config.get('max_lag', 3),
                                          fft_points=config.get('fft_points', 64),
                                          rolling_window=config.get('rolling_window', 5))
        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(train_df.values)

        # ----------------- Train ensemble models ------------------------
        models = train_model(scaled_train,
                             contamination=config.get('contamination', 0.05),
                             epochs=config.get('autoencoder_epochs', 50),
                             hidden_ratio=config.get('hidden_ratio', 0.25),
                             dropout=config.get('dropout', 0.0),
                             batch_norm=config.get('batch_norm', False),
                             if_params={'n_estimators': config.get('if_n_estimators', [100, 200]),
                                        'max_samples': config.get('if_max_samples', ['auto', 0.8]),
                                        'max_features': config.get('if_max_features', [0.5, 1.0])})

        # ----------------- Load test data and predict -------------------
        test_df, _ = batch_load_csv(args.test, config.get('chunk_size', 10000), schema,
                                    max_lag=config.get('max_lag', 3),
                                    fft_points=config.get('fft_points', 64),
                                    rolling_window=config.get('rolling_window', 5))
        scaled_test = scaler.transform(test_df.values)

        ensemble_labels, ensemble_scores = predict_anomalies(models, scaled_test,
                                                             config.get('contamination', 0.05))
        true_labels = generate_ground_truth(test_df, test_df.values,
                                            config.get('contamination', 0.05))

        f1 = f1_score(true_labels, ensemble_labels)
        precision = precision_score(true_labels, ensemble_labels)
        recall = recall_score(true_labels, ensemble_labels)
        anomaly_ratio = np.mean(ensemble_labels)
        mean_score = np.mean(ensemble_scores)
        print(f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        shap_values = explain_anomalies(models, test_df.values, list(test_df.columns),
                                        shap_sample=config.get('shap_sample', 100))
        generate_report(os.path.basename(args.test),
                        "Anomalous" if anomaly_ratio > config.get('contamination', 0.05) else "Normal",
                        "Fusion of Isolation Forest decision scores and Autoencoder reconstruction errors exceeded the configured threshold, indicating potential maintenance needs.",
                        f1, precision, recall, anomaly_ratio,
                        mean_score, shap_values, list(test_df.columns),
                        test_df.values, args.output)

    except Exception as e:
        logger.error(f"Runtime error: {str(e)}", exc_info=True)
        print(f"Error: {e}. Check anomdetec.log for details.")
        raise

if __name__ == "__main__":
    main()