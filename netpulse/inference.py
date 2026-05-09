"""
Inference module for NetPulse.

Handles real-time anomaly scoring of network flows using trained models.
Combines scores from Isolation Forest and Autoencoder for final alert decisions.
"""

import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.panel import Panel

from .features import FeatureProcessor, FEATURE_NAMES
from .models import Autoencoder, load_models


class AlertDatabase:
    """SQLite database for storing alert history."""
    
    def __init__(self, db_path: str = "~/.netpulse/alerts.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                src_ip TEXT NOT NULL,
                dst_ip TEXT NOT NULL,
                src_port INTEGER,
                dst_port INTEGER,
                protocol TEXT,
                combined_score REAL NOT NULL,
                if_score REAL,
                ae_score REAL,
                category TEXT,
                flow_duration_ms REAL,
                total_bytes_forward INTEGER,
                total_bytes_backward INTEGER
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON alerts(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_src_ip ON alerts(src_ip)
        """)
        
        conn.commit()
        conn.close()
    
    def add_alert(
        self,
        timestamp: float,
        src_ip: str,
        dst_ip: str,
        src_port: int,
        dst_port: int,
        protocol: str,
        combined_score: float,
        if_score: float,
        ae_score: float,
        category: str,
        flow_duration_ms: float = 0,
        total_bytes_forward: int = 0,
        total_bytes_backward: int = 0,
    ) -> int:
        """Add an alert to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO alerts (
                timestamp, src_ip, dst_ip, src_port, dst_port, protocol,
                combined_score, if_score, ae_score, category,
                flow_duration_ms, total_bytes_forward, total_bytes_backward
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, src_ip, dst_ip, src_port, dst_port, protocol,
            combined_score, if_score, ae_score, category,
            flow_duration_ms, total_bytes_forward, total_bytes_backward,
        ))
        
        alert_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return alert_id
    
    def get_alerts(
        self,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Retrieve alerts from the database."""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM alerts"
        params = []
        
        if since is not None:
            query += " WHERE timestamp >= ?"
            params.append(since)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def get_alert_count(self, since: Optional[float] = None) -> int:
        """Get total alert count."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if since is not None:
            cursor.execute("SELECT COUNT(*) FROM alerts WHERE timestamp >= ?", (since,))
        else:
            cursor.execute("SELECT COUNT(*) FROM alerts")
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count


class AnomalyDetector:
    """
    Real-time anomaly detection using trained models.
    
    Combines Isolation Forest and Autoencoder scores for robust detection.
    """
    
    def __init__(
        self,
        model_dir: Optional[Path] = None,
        combined_threshold: float = 1.0,
        if_weight: float = 0.5,
        ae_weight: float = 0.5,
    ):
        self.model_dir = model_dir
        self.combined_threshold = combined_threshold
        self.if_weight = if_weight
        self.ae_weight = ae_weight
        
        self.isolation_forest = None
        self.autoencoder = None
        self.feature_processor = FeatureProcessor()
        self.meta = None
        self.ae_threshold = 0.0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.console = Console()
        self.db = AlertDatabase()
        
        self._if_scores_min = 0.0
        self._if_scores_max = 1.0
        self._ae_errors_min = 0.0
        self._ae_errors_max = 1.0
    
    def load_models(self) -> None:
        """Load trained models from disk."""
        print(f"Loading models from {self.model_dir or 'current'}...")
        
        isolation_forest, autoencoder, meta = load_models(self.model_dir)
        
        self.isolation_forest = isolation_forest
        self.autoencoder = autoencoder.to(self.device)
        self.autoencoder.eval()
        self.meta = meta
        self.ae_threshold = meta.get("ae_threshold", 0.0)
        
        # Load scaler
        scaler_path = (self.model_dir or Path.home() / ".netpulse" / "models" / "current") / "scaler.pkl"
        self.feature_processor.load(str(scaler_path))
        
        print(f"Models loaded successfully.")
        print(f"  - Isolation Forest: {meta.get('n_samples', 0)} training samples")
        print(f"  - Autoencoder threshold: {self.ae_threshold:.6f}")
        print(f"  - Combined F1 score: {meta.get('metrics', {}).get('combined_f1', 0):.4f}")
    
    def score_flows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score flows for anomalies.
        
        Args:
            df: DataFrame with raw flow data
        
        Returns:
            DataFrame with added score columns
        """
        if self.isolation_forest is None or self.autoencoder is None:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        if df.empty:
            return df
        
        # Extract features
        features_df = self._extract_features(df)
        
        # Scale features
        X_scaled = self.feature_processor.transform(features_df)
        
        # Isolation Forest scores
        if_raw_scores = -self.isolation_forest.score_samples(X_scaled)
        
        # Update normalization bounds
        self._if_scores_min = min(self._if_scores_min, if_raw_scores.min())
        self._if_scores_max = max(self._if_scores_max, if_raw_scores.max())
        
        # Normalize IF scores
        if_range = self._if_scores_max - self._if_scores_min + 1e-10
        if_scores_norm = (if_raw_scores - self._if_scores_min) / if_range
        
        # Autoencoder reconstruction errors
        self.autoencoder.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            reconstructions = self.autoencoder(X_tensor)
            ae_errors = torch.mean((X_tensor - reconstructions) ** 2, dim=1).cpu().numpy()
        
        # Update normalization bounds
        self._ae_errors_min = min(self._ae_errors_min, ae_errors.min())
        self._ae_errors_max = max(self._ae_errors_max, ae_errors.max())
        
        # Normalize AE errors
        ae_range = self._ae_errors_max - self._ae_errors_min + 1e-10
        ae_errors_norm = (ae_errors - self._ae_errors_min) / ae_range
        
        # Combined score
        combined_scores = self.if_weight * if_scores_norm + self.ae_weight * ae_errors_norm
        
        # Add scores to dataframe
        result = df.copy()
        result["if_score"] = if_scores_norm
        result["ae_score"] = ae_errors_norm
        result["combined_score"] = combined_scores
        result["is_anomaly"] = combined_scores > self.combined_threshold
        
        return result
    
    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from flow data."""
        from .features import extract_flow_features
        
        features_df = extract_flow_features(df)
        
        # Ensure all required columns exist
        for col in FEATURE_NAMES:
            if col not in features_df.columns:
                features_df[col] = 0.0
        
        return features_df[FEATURE_NAMES]
    
    def categorize_anomaly(self, row: pd.Series) -> str:
        """Categorize the type of anomaly based on flow characteristics."""
        categories = []
        
        # High byte count = bandwidth spike
        if row.get("total_bytes_forward", 0) > 1_000_000 or row.get("total_bytes_backward", 0) > 1_000_000:
            categories.append("bandwidth_spike")
        
        # Many destination ports from same source = port scan
        if row.get("dst_port", 0) > 1000 and row.get("protocol") == "TCP":
            categories.append("potential_port_scan")
        
        # Unusual protocol
        if row.get("protocol") not in ["TCP", "UDP", "ICMP"]:
            categories.append("unusual_protocol")
        
        # High RST count = potential intrusion/reconnaissance
        if row.get("rst_count", 0) > 10:
            categories.append("high_rst_count")
        
        # SYN without ACK = potential SYN flood
        if row.get("syn_count", 0) > 20 and row.get("ack_count", 0) < 5:
            categories.append("potential_syn_flood")
        
        return ", ".join(categories) if categories else "unknown_anomaly"
    
    def process_window(
        self,
        flows_df: pd.DataFrame,
        alert_callback: Optional[callable] = None,
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Process a window of flows and generate alerts.
        
        Args:
            flows_df: DataFrame with flow data
            alert_callback: Optional callback function for alerts
        
        Returns:
            Tuple of (scored DataFrame, list of alert dicts)
        """
        if flows_df.empty:
            return flows_df, []
        
        # Score flows
        scored_df = self.score_flows(flows_df)
        
        # Generate alerts for anomalies
        alerts = []
        anomaly_rows = scored_df[scored_df["is_anomaly"]]
        
        for idx, row in anomaly_rows.iterrows():
            category = self.categorize_anomaly(row)
            
            alert = {
                "timestamp": time.time(),
                "src_ip": row.get("src_ip", "unknown"),
                "dst_ip": row.get("dst_ip", "unknown"),
                "src_port": row.get("src_port", 0),
                "dst_port": row.get("dst_port", 0),
                "protocol": row.get("protocol", "UNKNOWN"),
                "combined_score": row["combined_score"],
                "if_score": row["if_score"],
                "ae_score": row["ae_score"],
                "category": category,
                "flow_duration_ms": row.get("duration_ms", 0),
                "total_bytes_forward": row.get("total_bytes_forward", 0),
                "total_bytes_backward": row.get("total_bytes_backward", 0),
            }
            
            # Store in database
            self.db.add_alert(**alert)
            
            alerts.append(alert)
            
            # Call callback if provided
            if alert_callback:
                alert_callback(alert)
        
        return scored_df, alerts
    
    def render_alert(self, alert: Dict) -> None:
        """Render an alert to the terminal using rich."""
        color = "red" if alert["combined_score"] > 1.5 else "yellow"
        
        panel = Panel(
            f"[bold]{alert['category']}[/bold]\n\n"
            f"Source: [cyan]{alert['src_ip']}[/cyan] → "
            f"Dest: [cyan]{alert['dst_ip']}:{alert['dst_port']}[/cyan]\n"
            f"Protocol: {alert['protocol']}\n\n"
            f"Combined Score: [{color}]{alert['combined_score']:.3f}[/{color}]\n"
            f"IF Score: {alert['if_score']:.3f} | "
            f"AE Score: {alert['ae_score']:.3f}\n\n"
            f"Time: {datetime.fromtimestamp(alert['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}",
            title=f"[bold red]⚠ ANOMALY DETECTED[/bold red]",
            border_style=color,
        )
        
        self.console.print(panel)
    
    def get_status(self) -> Dict:
        """Get current detector status."""
        return {
            "models_loaded": self.isolation_forest is not None,
            "combined_threshold": self.combined_threshold,
            "device": str(self.device),
            "total_alerts": self.db.get_alert_count(),
            "model_metrics": self.meta.get("metrics", {}) if self.meta else {},
        }
