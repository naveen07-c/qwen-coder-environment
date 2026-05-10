# NetPulse

**Intelligent Network Anomaly Detector**

NetPulse is a CLI-based intelligent network anomaly detector that passively monitors your local network traffic, learns what "normal" looks like using unsupervised machine learning, and raises real-time alerts when something deviates — rogue devices, port scans, bandwidth spikes, unusual protocols, or potential intrusions. It runs entirely offline, costs nothing, and improves over time through nightly retraining.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- **Passive Monitoring**: Captures network traffic without disrupting normal operations
- **Unsupervised ML**: Uses Isolation Forest and Autoencoder models to detect anomalies
- **Real-time Alerts**: Instant notifications via terminal, SQLite database, and webhooks
- **Self-Improving**: Nightly retraining adapts to your network's evolving patterns
- **Privacy-First**: All processing happens locally; no data leaves your machine
- **Zero Cost**: No subscriptions, APIs, or cloud services required

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/netpulse.git
cd netpulse

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install NetPulse
pip install -e .
```

### Configuration

```bash
# Create configuration directory
mkdir -p ~/.netpulse

# Copy default config
cp netpulse/config.toml.example ~/.netpulse/config.toml

# Edit configuration (optional)
netpulse config
```

### Usage

```bash
# Start monitoring (requires sudo for packet capture)
sudo netpulse start

# Start monitoring on specific interface
sudo netpulse start --interface eth0

# View live dashboard
netpulse status

# Generate alert report
netpulse report --last 24h

# Export alerts to CSV
netpulse report --last 7d --export alerts.csv

# Manually trigger model training
netpulse train

# View evaluation metrics
netpulse eval

# Reset baseline and start fresh
netpulse reset-baseline
```

## Architecture

NetPulse processes network traffic through a multi-stage pipeline:

1. **Packet Capture**: Uses `scapy` to capture Layer 3/4 packets in promiscuous mode
2. **Flow Aggregation**: Groups packets into 5-tuple flows (src_ip, dst_ip, src_port, dst_port, protocol)
3. **Feature Extraction**: Computes 18 features per flow including byte counts, packet sizes, inter-arrival times, TCP flags, and port entropy
4. **Preprocessing**: Log-scales heavy-tailed features and normalizes to [0, 1] using MinMaxScaler
5. **Anomaly Detection**: 
   - Isolation Forest: Identifies outliers through random partitioning
   - Autoencoder: Detects anomalies via reconstruction error
6. **Alert Generation**: Combined weighted score triggers alerts when threshold exceeded

## Model Details

### Isolation Forest
- 200 estimators
- Contamination: 0.01
- Works by isolating anomalies through random binary splits

### Autoencoder
- Architecture: 18 → 64 → 32 → 8 → 32 → 64 → 18
- Trained with MSE loss on normal traffic
- Anomalies produce high reconstruction error

### Combined Scoring
```
combined_score = 0.5 × IF_score + 0.5 × AE_reconstruction_error
alert_triggered = combined_score > threshold
```

## Project Structure

```
netpulse/
├── __init__.py          # Package initialization
├── __main__.py          # CLI entry point
├── capture.py           # Packet capture & flow aggregation
├── cli.py               # Command-line interface
├── config.py            # Configuration management
├── data.py              # Dataset loaders (CICIDS2018, UNSW-NB15)
├── features.py          # Feature engineering (18 features)
├── inference.py         # Real-time scoring & alerting
├── models.py            # Model training (IF + Autoencoder)
├── retrain.py           # Nightly retraining scheduler
├── requirements.txt     # Python dependencies
├── setup.py             # Package installation script
└── config.toml.example  # Default configuration
```

## Runtime Data

NetPulse stores runtime data in `~/.netpulse/`:

```
~/.netpulse/
├── models/
│   ├── current/              # Symlink to latest validated version
│   └── vYYYY-MM-DD/          # Versioned model directories
│       ├── isolation_forest.pkl
│       ├── autoencoder.pt
│       ├── scaler.pkl
│       └── model_meta.json
├── data/
│   └── traffic_log.parquet   # Rolling 7-day traffic window
├── alerts.db                 # SQLite alert history
├── reports/                  # Generated reports
├── eval_history.json         # Evaluation metrics history
└── config.toml               # User configuration
```

## Requirements

- Python 3.8+
- Linux/macOS (Windows support limited)
- Root/sudo access for packet capture
- 8GB RAM minimum
- No GPU required (but speeds up training)

### Dependencies

All dependencies are listed in `requirements.txt`:
- scapy (packet capture)
- pandas, numpy (data processing)
- scikit-learn (Isolation Forest, preprocessing)
- torch (Autoencoder)
- click, rich (CLI)
- sqlite3 (alert storage)
- schedule (retraining)

## Training Data

NetPulse can be trained on:
1. **CICIDS2018**: 16M labeled flows from Canadian Institute for Cybersecurity
2. **UNSW-NB15**: 2.5M flows from UNSW Sydney
3. **Personal Baseline**: 7-day passive capture of your own network

## Performance Benchmarks

Based on CICIDS2018 evaluation:
- Precision: ≥ 0.85
- Recall: ≥ 0.80
- F1-Score: ≥ 0.82
- AUC-ROC: ≥ 0.90
- Inference latency: < 10ms per 30-second window

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Disclaimer

NetPulse is a security research tool. Use responsibly and only on networks you own or have explicit permission to monitor. This tool is provided "as is" without warranty of any kind.

## Acknowledgments

- CICIDS2018 dataset: Canadian Institute for Cybersecurity
- UNSW-NB15 dataset: UNSW Sydney
- Built with open-source tools and libraries
