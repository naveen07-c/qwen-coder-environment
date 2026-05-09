"""
Configuration management for NetPulse.

Handles loading and saving configuration from/to TOML files.
"""

import os
import toml
from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG = {
    "general": {
        "interface": "",
        "mode": "capture",
        "scoring_window": 30,
        "buffer_duration": 5,
    },
    "models": {
        "isolation_forest_contamination": 0.01,
        "autoencoder_bottleneck": 8,
        "autoencoder_epochs": 30,
        "autoencoder_batch_size": 512,
        "autoencoder_learning_rate": 0.001,
        "combined_threshold": 1.0,
        "if_weight": 0.5,
        "ae_weight": 0.5,
    },
    "retraining": {
        "retrain_hour": 2,
        "retrain_minute": 0,
        "max_f1_regression": 0.05,
        "retrain_window_days": 7,
    },
    "alerts": {
        "db_path": "~/.netpulse/alerts.db",
        "webhook_url": "",
        "terminal_alerts": True,
    },
    "logging": {
        "level": "INFO",
        "log_path": "~/.netpulse/netpulse.log",
    },
}


def get_netpulse_dir() -> Path:
    """Get the NetPulse configuration directory."""
    return Path.home() / ".netpulse"


def get_config_path() -> Path:
    """Get the path to the configuration file."""
    return get_netpulse_dir() / "config.toml"


def ensure_config_dir() -> None:
    """Ensure the configuration directory exists."""
    get_netpulse_dir().mkdir(parents=True, exist_ok=True)
    (get_netpulse_dir() / "models").mkdir(exist_ok=True)
    (get_netpulse_dir() / "data").mkdir(exist_ok=True)
    (get_netpulse_dir() / "reports").mkdir(exist_ok=True)
    (get_netpulse_dir() / "models" / "current").mkdir(exist_ok=True)


def load_config() -> Dict[str, Any]:
    """Load configuration from file, merging with defaults."""
    ensure_config_dir()
    config_path = get_config_path()
    
    if config_path.exists():
        with open(config_path, "r") as f:
            file_config = toml.load(f)
        
        # Merge with defaults
        config = DEFAULT_CONFIG.copy()
        for section, values in file_config.items():
            if section in config:
                config[section].update(values)
            else:
                config[section] = values
        return config
    else:
        # Create default config file
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()


def save_config(config: Dict[str, Any]) -> None:
    """Save configuration to file."""
    ensure_config_dir()
    config_path = get_config_path()
    
    with open(config_path, "w") as f:
        toml.dump(config, f)


def get_config_value(section: str, key: str, default: Any = None) -> Any:
    """Get a specific configuration value."""
    config = load_config()
    return config.get(section, {}).get(key, default)


def update_config_value(section: str, key: str, value: Any) -> None:
    """Update a specific configuration value."""
    config = load_config()
    if section not in config:
        config[section] = {}
    config[section][key] = value
    save_config(config)
