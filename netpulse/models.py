"""
Model training module for NetPulse.

Implements two complementary anomaly detection models:
- Model A: Isolation Forest (scikit-learn)
- Model B: Autoencoder (PyTorch)
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import joblib
import matplotlib.pyplot as plt


class Autoencoder(nn.Module):
    """
    Fully connected autoencoder for anomaly detection.
    
    Architecture: 18 → 64 → 32 → 8 → 32 → 64 → 18
    The encoder compresses input to 8-dimensional latent representation.
    """
    
    def __init__(self, input_dim: int = 18, bottleneck_dim: int = 8):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, bottleneck_dim),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Get the latent representation."""
        return self.encoder(x)


class ModelTrainer:
    """
    Handles training of both Isolation Forest and Autoencoder models.
    """
    
    def __init__(
        self,
        if_contamination: float = 0.01,
        ae_bottleneck: int = 8,
        ae_epochs: int = 30,
        ae_batch_size: int = 512,
        ae_learning_rate: float = 1e-3,
        random_state: int = 42,
    ):
        self.if_contamination = if_contamination
        self.ae_bottleneck = ae_bottleneck
        self.ae_epochs = ae_epochs
        self.ae_batch_size = ae_batch_size
        self.ae_learning_rate = ae_learning_rate
        self.random_state = random_state
        
        self.isolation_forest: Optional[IsolationForest] = None
        self.autoencoder: Optional[Autoencoder] = None
        self.ae_threshold: float = 0.0  # 95th percentile reconstruction error
    
    def train_isolation_forest(self, X_train: np.ndarray) -> IsolationForest:
        """
        Train Isolation Forest model.
        
        Args:
            X_train: Training features scaled to [0, 1]
        
        Returns:
            Trained IsolationForest model
        """
        print("Training Isolation Forest...")
        start_time = time.time()
        
        self.isolation_forest = IsolationForest(
            n_estimators=200,
            contamination=self.if_contamination,
            max_samples="auto",
            random_state=self.random_state,
            n_jobs=-1,
        )
        
        self.isolation_forest.fit(X_train)
        
        elapsed = time.time() - start_time
        print(f"Isolation Forest training completed in {elapsed:.2f}s")
        
        return self.isolation_forest
    
    def train_autoencoder(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
    ) -> Autoencoder:
        """
        Train Autoencoder model.
        
        Args:
            X_train: Training features scaled to [0, 1]
            X_val: Validation features for early stopping
        
        Returns:
            Trained Autoencoder model
        """
        print("Training Autoencoder...")
        start_time = time.time()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Create model
        self.autoencoder = Autoencoder(
            input_dim=X_train.shape[1],
            bottleneck_dim=self.ae_bottleneck,
        ).to(device)
        
        # Prepare data
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.ae_batch_size,
            shuffle=True,
        )
        
        # Validation data
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(device)
            val_dataset = TensorDataset(X_val_tensor, X_val_tensor)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.ae_batch_size,
                shuffle=False,
            )
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.autoencoder.parameters(),
            lr=self.ae_learning_rate,
        )
        
        # Training loop
        best_val_loss = float("inf")
        patience = 5
        patience_counter = 0
        
        for epoch in range(self.ae_epochs):
            # Training
            self.autoencoder.train()
            train_loss = 0.0
            for batch_x, _ in train_loader:
                optimizer.zero_grad()
                outputs = self.autoencoder(batch_x)
                loss = criterion(outputs, batch_x)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            val_loss = 0.0
            if X_val is not None:
                self.autoencoder.eval()
                with torch.no_grad():
                    for batch_x, _ in val_loader:
                        outputs = self.autoencoder(batch_x)
                        loss = criterion(outputs, batch_x)
                        val_loss += loss.item()
                val_loss /= len(val_loader)
                
                print(f"Epoch {epoch+1}/{self.ae_epochs} - "
                      f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"Epoch {epoch+1}/{self.ae_epochs} - Train Loss: {train_loss:.6f}")
        
        # Compute threshold on validation set (95th percentile reconstruction error)
        self.autoencoder.eval()
        with torch.no_grad():
            X_train_tensor = torch.FloatTensor(X_train).to(device)
            reconstructions = self.autoencoder(X_train_tensor)
            errors = torch.mean((X_train_tensor - reconstructions) ** 2, dim=1)
            self.ae_threshold = torch.quantile(errors, 0.95).item()
        
        elapsed = time.time() - start_time
        print(f"Autoencoder training completed in {elapsed:.2f}s")
        print(f"Reconstruction error threshold (95th percentile): {self.ae_threshold:.6f}")
        
        return self.autoencoder
    
    def evaluate(
        self,
        X_attack: np.ndarray,
        y_attack: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate models on attack data.
        
        Args:
            X_attack: Attack features scaled to [0, 1]
            y_attack: Binary labels (1 = attack, 0 = normal)
        
        Returns:
            Dictionary of evaluation metrics
        """
        if self.isolation_forest is None or self.autoencoder is None:
            raise ValueError("Models must be trained before evaluation")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Isolation Forest predictions
        # score_samples returns negative anomaly scores (more negative = more anomalous)
        if_scores = -self.isolation_forest.score_samples(X_attack)
        if_predictions = (if_scores > np.percentile(if_scores, 99)).astype(int)
        
        # Autoencoder predictions
        self.autoencoder.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_attack).to(device)
            reconstructions = self.autoencoder(X_tensor)
            ae_errors = torch.mean((X_tensor - reconstructions) ** 2, dim=1).cpu().numpy()
        
        # Normalize scores to [0, 1]
        if_scores_norm = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-10)
        ae_errors_norm = (ae_errors - ae_errors.min()) / (ae_errors.max() - ae_errors.min() + 1e-10)
        
        # Combined score
        combined_scores = 0.5 * if_scores_norm + 0.5 * ae_errors_norm
        combined_predictions = (combined_scores > 0.5).astype(int)
        
        # Compute metrics
        metrics = {
            "if_precision": precision_score(y_attack, if_predictions),
            "if_recall": recall_score(y_attack, if_predictions),
            "if_f1": f1_score(y_attack, if_predictions),
            "if_auc": roc_auc_score(y_attack, if_scores_norm),
            "ae_precision": precision_score(y_attack, (ae_errors_norm > 0.5).astype(int)),
            "ae_recall": recall_score(y_attack, (ae_errors_norm > 0.5).astype(int)),
            "ae_f1": f1_score(y_attack, (ae_errors_norm > 0.5).astype(int)),
            "ae_auc": roc_auc_score(y_attack, ae_errors_norm),
            "combined_precision": precision_score(y_attack, combined_predictions),
            "combined_recall": recall_score(y_attack, combined_predictions),
            "combined_f1": f1_score(y_attack, combined_predictions),
            "combined_auc": roc_auc_score(y_attack, combined_scores),
        }
        
        return metrics
    
    def plot_roc_curve(
        self,
        X_attack: np.ndarray,
        y_attack: np.ndarray,
        save_path: str,
    ) -> None:
        """Plot and save ROC curve."""
        from sklearn.metrics import RocCurveDisplay
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Isolation Forest
        if_scores = -self.isolation_forest.score_samples(X_attack)
        if_scores_norm = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-10)
        RocCurveDisplay.from_predictions(y_attack, if_scores_norm, name="Isolation Forest", ax=ax)
        
        # Autoencoder
        self.autoencoder.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_attack).to(device)
            reconstructions = self.autoencoder(X_tensor)
            ae_errors = torch.mean((X_tensor - reconstructions) ** 2, dim=1).cpu().numpy()
        ae_errors_norm = (ae_errors - ae_errors.min()) / (ae_errors.max() - ae_errors.min() + 1e-10)
        RocCurveDisplay.from_predictions(y_attack, ae_errors_norm, name="Autoencoder", ax=ax)
        
        ax.set_title("ROC Curve - NetPulse Anomaly Detection")
        ax.grid(True, alpha=0.3)
        
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"ROC curve saved to {save_path}")


def get_model_dir(version: Optional[str] = None) -> Path:
    """Get the model directory path."""
    base_dir = Path.home() / ".netpulse" / "models"
    if version is None:
        version = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return base_dir / f"v{version}"


def save_models(
    trainer: ModelTrainer,
    feature_processor,
    metrics: Dict[str, float],
    n_samples: int,
    version: Optional[str] = None,
) -> Path:
    """Save trained models and metadata to disk."""
    model_dir = get_model_dir(version)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().isoformat()
    
    # Save Isolation Forest
    if_path = model_dir / "isolation_forest.pkl"
    joblib.dump(trainer.isolation_forest, if_path)
    
    # Save Autoencoder
    ae_path = model_dir / "autoencoder.pt"
    torch.save(trainer.autoencoder.state_dict(), ae_path)
    
    # Save scaler
    scaler_path = model_dir / "scaler.pkl"
    feature_processor.save(scaler_path)
    
    # Save metadata
    meta = {
        "version": version or timestamp,
        "training_date": timestamp,
        "n_samples": n_samples,
        "metrics": metrics,
        "if_contamination": trainer.if_contamination,
        "ae_bottleneck": trainer.ae_bottleneck,
        "ae_threshold": trainer.ae_threshold,
    }
    meta_path = model_dir / "model_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    # Update current symlink
    current_dir = Path.home() / ".netpulse" / "models" / "current"
    if current_dir.exists() or current_dir.is_symlink():
        current_dir.unlink()
    current_dir.symlink_to(model_dir)
    
    print(f"Models saved to {model_dir}")
    print(f"Current symlink updated to point to v{version or 'latest'}")
    
    return model_dir


def load_models(model_dir: Optional[Path] = None) -> Tuple:
    """Load models from disk."""
    if model_dir is None:
        model_dir = Path.home() / ".netpulse" / "models" / "current"
        if not model_dir.exists():
            raise FileNotFoundError("No current model found. Run training first.")
    
    # Load Isolation Forest
    if_path = model_dir / "isolation_forest.pkl"
    isolation_forest = joblib.load(if_path)
    
    # Load Autoencoder
    ae_path = model_dir / "autoencoder.pt"
    autoencoder = Autoencoder()
    autoencoder.load_state_dict(torch.load(ae_path))
    autoencoder.eval()
    
    # Load metadata
    meta_path = model_dir / "model_meta.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    return isolation_forest, autoencoder, meta
