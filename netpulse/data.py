"""
Data loader module for NetPulse.

Handles loading and preprocessing of public datasets (CICIDS2018, UNSW-NB15)
for model training and evaluation.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


def load_cicids2018(
    data_dir: str,
    normal_only: bool = False,
    chunksize: int = 100000,
) -> pd.DataFrame:
    """
    Load CICIDS2018 dataset.
    
    Args:
        data_dir: Directory containing CICIDS2018 CSV files
        normal_only: If True, only load normal traffic
        chunksize: Number of rows to process at a time
    
    Returns:
        DataFrame with network flows and labels
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"CICIDS2018 data directory not found: {data_dir}")
    
    # CICIDS2018 file mapping (adjust based on actual file names)
    files = list(data_path.glob("*.csv"))
    
    if not files:
        raise FileNotFoundError("No CSV files found in CICIDS2018 directory")
    
    print(f"Found {len(files)} CICIDS2018 files")
    
    all_data = []
    
    for file_path in files:
        print(f"Processing {file_path.name}...")
        
        # Read in chunks to handle large files
        chunks = []
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            if normal_only:
                # Filter for normal traffic only
                # Label column name may vary - adjust as needed
                label_col = None
                for col in ["Label", "label", "Attack", "attack"]:
                    if col in chunk.columns:
                        label_col = col
                        break
                
                if label_col and "BENIGN" in chunk[label_col].values:
                    chunk = chunk[chunk[label_col] == "BENIGN"]
            
            if len(chunk) > 0:
                chunks.append(chunk)
        
        if chunks:
            file_df = pd.concat(chunks, ignore_index=True)
            all_data.append(file_df)
            print(f"  Loaded {len(file_df)} rows")
    
    if not all_data:
        return pd.DataFrame()
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"Total: {len(combined)} rows")
    
    return combined


def load_unsw_nb15(
    data_dir: str,
    normal_only: bool = False,
) -> pd.DataFrame:
    """
    Load UNSW-NB15 dataset.
    
    Args:
        data_dir: Directory containing UNSW-NB15 CSV files
        normal_only: If True, only load normal traffic
    
    Returns:
        DataFrame with network flows and labels
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"UNSW-NB15 data directory not found: {data_dir}")
    
    files = list(data_path.glob("*.csv"))
    
    if not files:
        raise FileNotFoundError("No CSV files found in UNSW-NB15 directory")
    
    all_data = []
    
    for file_path in files:
        print(f"Processing {file_path.name}...")
        
        df = pd.read_csv(file_path)
        
        if normal_only:
            # Filter for normal traffic
            # Label column is typically 'label' or 'Label'
            label_col = None
            for col in ["label", "Label", "attack_cat", "AttackCat"]:
                if col in df.columns:
                    label_col = col
                    break
            
            if label_col:
                # Normal traffic typically labeled as 0 or "Normal"
                if df[label_col].dtype == object:
                    df = df[df[label_col].str.lower() == "normal"]
                else:
                    df = df[df[label_col] == 0]
        
        if len(df) > 0:
            all_data.append(df)
            print(f"  Loaded {len(df)} rows")
    
    if not all_data:
        return pd.DataFrame()
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"Total: {len(combined)} rows")
    
    return combined


def map_cicids_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map CICIDS2018 features to NetPulse feature set.
    
    CICIDS2018 has different column names than our 18-feature set.
    This function maps the relevant columns.
    
    Args:
        features_df: Raw CICIDS2018 DataFrame
    
    Returns:
        DataFrame with mapped features
    """
    # CICIDS2018 column name mapping (approximate - adjust based on actual data)
    column_mapping = {
        "Flow Duration": "flow_duration_ms",
        "Fwd Total Length of Packets": "total_forward_bytes",
        "Bwd Total Length of Packets": "total_backward_bytes",
        "Fwd Packet Length Max": "mean_forward_packet_size",  # Approximation
        "Bwd Packet Length Max": "mean_backward_packet_size",  # Approximation
        "Fwd Packets/s": "forward_packet_count",  # Approximation
        "Bwd Packets/s": "backward_packet_count",  # Approximation
        "Fwd IAT Mean": "mean_forward_inter_arrival",
        "Fwd IAT Std": "std_forward_inter_arrival",
        "Bwd IAT Mean": "mean_backward_inter_arrival",
        "Bwd IAT Std": "std_backward_inter_arrival",
        "Dst Port": "dst_port_normalized",
    }
    
    mapped = pd.DataFrame()
    
    for cic_col, netpulse_col in column_mapping.items():
        if cic_col in features_df.columns:
            mapped[netpulse_col] = features_df[cic_col]
    
    # Add protocol one-hot if available
    if "Protocol" in features_df.columns:
        mapped["protocol_tcp"] = (features_df["Protocol"] == 6).astype(float)
        mapped["protocol_udp"] = (features_df["Protocol"] == 17).astype(float)
        mapped["protocol_icmp"] = (features_df["Protocol"] == 1).astype(float)
    else:
        mapped["protocol_tcp"] = 1.0
        mapped["protocol_udp"] = 0.0
        mapped["protocol_icmp"] = 0.0
    
    # Add TCP flags if available
    flag_mapping = {
        "Fwd URG Flags Count": None,
        "Fwd SYN Flag Count": "syn_count",
        "Fwd ACK Flag Count": "ack_count",
        "Fwd FIN Flag Count": "fin_count",
        "Fwd RST Flag Count": "rst_count",
    }
    
    for cic_col, netpulse_col in flag_mapping.items():
        if netpulse_col and cic_col in features_df.columns:
            mapped[netpulse_col] = features_df[cic_col]
        elif netpulse_col:
            mapped[netpulse_col] = 0
    
    return mapped


def prepare_training_data(
    cicids_dir: Optional[str] = None,
    unsw_dir: Optional[str] = None,
    personal_data_path: Optional[str] = None,
    test_split: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare training and test data from multiple sources.
    
    Args:
        cicids_dir: Path to CICIDS2018 dataset
        unsw_dir: Path to UNSW-NB15 dataset
        personal_data_path: Path to personal traffic capture
        test_split: Fraction of data to use for validation
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    all_normal = []
    all_attack = []
    
    # Load CICIDS2018
    if cicids_dir:
        print("Loading CICIDS2018...")
        cic_normal = load_cicids2018(cicids_dir, normal_only=True)
        cic_attack = load_cicids2018(cicids_dir, normal_only=False)
        
        if len(cic_normal) > 0:
            all_normal.append(cic_normal)
        
        # Separate attack traffic
        if len(cic_attack) > 0:
            # Assuming there's a Label column
            label_col = None
            for col in ["Label", "label"]:
                if col in cic_attack.columns:
                    label_col = col
                    break
            
            if label_col:
                attacks = cic_attack[cic_attack[label_col] != "BENIGN"]
                if len(attacks) > 0:
                    all_attack.append(attacks)
    
    # Load UNSW-NB15
    if unsw_dir:
        print("Loading UNSW-NB15...")
        unsw_normal = load_unsw_nb15(unsw_dir, normal_only=True)
        unsw_attack = load_unsw_nb15(unsw_dir, normal_only=False)
        
        if len(unsw_normal) > 0:
            all_normal.append(unsw_normal)
        
        if len(unsw_attack) > 0:
            all_attack.append(unsw_attack)
    
    # Load personal data
    if personal_data_path and Path(personal_data_path).exists():
        print("Loading personal traffic data...")
        personal = pd.read_parquet(personal_data_path)
        if len(personal) > 0:
            all_normal.append(personal)
    
    if not all_normal:
        raise ValueError("No training data found. Provide at least one data source.")
    
    # Combine all normal traffic
    normal_df = pd.concat(all_normal, ignore_index=True)
    print(f"Total normal samples: {len(normal_df)}")
    
    # Split normal data into train/val
    train_df, val_df = train_test_split(
        normal_df,
        test_size=test_split,
        random_state=random_state,
    )
    
    # Combine attack data for testing
    if all_attack:
        attack_df = pd.concat(all_attack, ignore_index=True)
        print(f"Total attack samples: {len(attack_df)}")
    else:
        attack_df = pd.DataFrame()
    
    return train_df, val_df, attack_df, normal_df


def save_eval_history(
    metrics: Dict,
    history_path: Optional[str] = None,
) -> None:
    """
    Save evaluation metrics to history file.
    
    Args:
        metrics: Dictionary of evaluation metrics
        history_path: Path to history JSON file
    """
    import json
    from datetime import datetime
    
    if history_path is None:
        history_path = Path.home() / ".netpulse" / "eval_history.json"
    
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing history
    if history_path.exists():
        with open(history_path, "r") as f:
            history = json.load(f)
    else:
        history = []
    
    # Add new entry
    entry = {
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
    }
    history.append(entry)
    
    # Keep only last 100 entries
    history = history[-100:]
    
    # Save
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"Evaluation history saved to {history_path}")
