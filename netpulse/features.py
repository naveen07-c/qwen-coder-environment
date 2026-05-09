"""
Feature engineering module for NetPulse.

Extracts 18 features from flow records that distinguish normal from anomalous traffic.
Applies preprocessing steps including log-scaling and normalization.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from sklearn.preprocessing import MinMaxScaler
import joblib


# Feature names in order
FEATURE_NAMES = [
    "flow_duration_ms",
    "total_forward_bytes",
    "total_backward_bytes",
    "forward_packet_count",
    "backward_packet_count",
    "mean_forward_packet_size",
    "mean_backward_packet_size",
    "mean_forward_inter_arrival",
    "std_forward_inter_arrival",
    "mean_backward_inter_arrival",
    "std_backward_inter_arrival",
    "syn_count",
    "ack_count",
    "fin_count",
    "rst_count",
    "dst_port_normalized",
    "protocol_tcp",
    "protocol_udp",
    "protocol_icmp",
]


def compute_shannon_entropy(values: List[int]) -> float:
    """Compute Shannon entropy for a list of values (e.g., destination ports)."""
    if not values:
        return 0.0
    
    # Count frequencies
    unique, counts = np.unique(values, return_counts=True)
    probabilities = counts / len(values)
    
    # Compute entropy
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    return entropy


def extract_flow_features(df: pd.DataFrame, port_entropy_map: Optional[dict] = None) -> pd.DataFrame:
    """
    Extract 18 features from flow records.
    
    Args:
        df: DataFrame with raw flow data from capture module
        port_entropy_map: Optional dict mapping src_ip to port entropy
    
    Returns:
        DataFrame with extracted features
    """
    features = []
    
    for idx, row in df.iterrows():
        # Handle empty lists safely
        forward_sizes = row.get("forward_sizes", [])
        backward_sizes = row.get("backward_sizes", [])
        forward_iat = row.get("forward_inter_arrival", [])
        backward_iat = row.get("backward_inter_arrival", [])
        
        # Ensure lists are actually lists
        if not isinstance(forward_sizes, list):
            forward_sizes = []
        if not isinstance(backward_sizes, list):
            backward_sizes = []
        if not isinstance(forward_iat, list):
            forward_iat = []
        if not isinstance(backward_iat, list):
            backward_iat = []
        
        feature_row = {
            # Flow duration
            "flow_duration_ms": max(0, row.get("duration_ms", 0)),
            
            # Log-scaled byte counts
            "total_forward_bytes": np.log1p(row.get("total_bytes_forward", 0)),
            "total_backward_bytes": np.log1p(row.get("total_bytes_backward", 0)),
            
            # Packet counts
            "forward_packet_count": max(0, row.get("packet_count_forward", 0)),
            "backward_packet_count": max(0, row.get("packet_count_backward", 0)),
            
            # Mean packet sizes
            "mean_forward_packet_size": np.mean(forward_sizes) if forward_sizes else 0,
            "mean_backward_packet_size": np.mean(backward_sizes) if backward_sizes else 0,
            
            # Inter-arrival time statistics (forward)
            "mean_forward_inter_arrival": np.mean(forward_iat) if forward_iat else 0,
            "std_forward_inter_arrival": np.std(forward_iat) if len(forward_iat) > 1 else 0,
            
            # Inter-arrival time statistics (backward)
            "mean_backward_inter_arrival": np.mean(backward_iat) if backward_iat else 0,
            "std_backward_inter_arrival": np.std(backward_iat) if len(backward_iat) > 1 else 0,
            
            # TCP flag counts
            "syn_count": max(0, row.get("syn_count", 0)),
            "ack_count": max(0, row.get("ack_count", 0)),
            "fin_count": max(0, row.get("fin_count", 0)),
            "rst_count": max(0, row.get("rst_count", 0)),
            
            # Destination port (normalized to [0, 1] by dividing by max port 65535)
            "dst_port_normalized": row.get("dst_port", 0) / 65535.0,
            
            # Protocol one-hot encoding
            "protocol_tcp": 1.0 if row.get("protocol", "") == "TCP" else 0.0,
            "protocol_udp": 1.0 if row.get("protocol", "") == "UDP" else 0.0,
            "protocol_icmp": 1.0 if row.get("protocol", "") == "ICMP" else 0.0,
        }
        
        features.append(feature_row)
    
    features_df = pd.DataFrame(features)
    
    # Add port entropy if provided
    if port_entropy_map is not None and "src_ip" in df.columns:
        features_df["port_entropy"] = df["src_ip"].apply(
            lambda ip: port_entropy_map.get(ip, 0.0)
        )
    
    return features_df


def compute_port_entropy(df: pd.DataFrame, window_seconds: float = 60.0) -> dict:
    """
    Compute destination port entropy per source IP within a time window.
    
    High entropy indicates a device contacting many different ports (potential port scan).
    
    Args:
        df: DataFrame with flow data including src_ip, dst_port, start_time
        window_seconds: Time window for computing entropy
    
    Returns:
        Dict mapping src_ip to port entropy value
    """
    if df.empty or "src_ip" not in df.columns:
        return {}
    
    entropy_map = {}
    now = time.time() if "start_time" not in df.columns else df["start_time"].max()
    
    # Filter to recent flows
    if "start_time" in df.columns:
        recent_df = df[df["start_time"] >= (now - window_seconds)]
    else:
        recent_df = df
    
    # Group by source IP and compute entropy
    for src_ip in recent_df["src_ip"].unique():
        ports = recent_df[recent_df["src_ip"] == src_ip]["dst_port"].tolist()
        entropy_map[src_ip] = compute_shannon_entropy(ports)
    
    return entropy_map


class FeatureProcessor:
    """
    Handles feature extraction and preprocessing pipeline.
    
    Fits scalers on training data and applies the same transformation at inference.
    """
    
    def __init__(self):
        self.scaler: Optional[MinMaxScaler] = None
        self.feature_columns = FEATURE_NAMES.copy()
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> "FeatureProcessor":
        """
        Fit the scaler on training data.
        
        Args:
            df: DataFrame with raw features (columns matching FEATURE_NAMES)
        
        Returns:
            self
        """
        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0
        
        # Extract only the feature columns
        X = df[self.feature_columns].values
        
        # Fit scaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit(X)
        self.is_fitted = True
        
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted scaler.
        
        Args:
            df: DataFrame with raw features
        
        Returns:
            Scaled feature array
        """
        if not self.is_fitted:
            raise ValueError("FeatureProcessor must be fitted before transform")
        
        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0
        
        X = df[self.feature_columns].values
        return self.scaler.transform(X)
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data back to original scale."""
        if not self.is_fitted:
            raise ValueError("FeatureProcessor must be fitted before inverse_transform")
        return self.scaler.inverse_transform(X)
    
    def save(self, filepath: str) -> None:
        """Save the fitted scaler to disk."""
        joblib.dump(self.scaler, filepath)
    
    def load(self, filepath: str) -> None:
        """Load a fitted scaler from disk."""
        self.scaler = joblib.load(filepath)
        self.is_fitted = True


import time
