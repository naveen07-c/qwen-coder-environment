# NetPulse

**Intelligent Network Anomaly Detector**

NetPulse is a CLI-based intelligent network anomaly detector that passively monitors your local network traffic, learns what "normal" looks like using unsupervised ML, and raises real-time alerts when something deviates.

## Features

- 🛡️ **Unsupervised Anomaly Detection** - Uses Isolation Forest and Autoencoder models to detect deviations from normal traffic patterns
- 📊 **Real-time Monitoring** - 30-second scoring windows with instant alerting
- 🧠 **Self-Improving** - Nightly retraining keeps the baseline fresh as your network evolves
- 💻 **Offline & Private** - Runs entirely on your machine, no cloud services
- 🎯 **Multiple Attack Detection** - Port scans, bandwidth spikes, SYN floods, unusual protocols, and more

## Installation

```bash
# Clone the repository
git clone https://github.com/netpulse/netpulse.git
cd netpulse

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### System Requirements

- Python 3.9+
- 8GB RAM minimum (16GB recommended)
- Linux or macOS (requires sudo for packet capture)
- No GPU required (but supported for faster training)

## Quick Start

### 1. Baseline Collection (7 days recommended)

```bash
sudo netpulse start --mode capture
```

This collects your personal network traffic to build a customized baseline.

### 2. Train Models

```bash
netpulse train
```

Trains both Isolation Forest and Autoencoder models on your captured traffic.

### 3. Start Monitoring

```bash
sudo netpulse start --mode alert
```

Begin real-time anomaly detection with alerts.

### 4. View Status

```bash
netpulse status      # Live dashboard
netpulse report      # Alert summary
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `netpulse start` | Begin monitoring (requires sudo) |
| `netpulse start -i eth0` | Specify network interface |
| `netpulse status` | Live anomaly score dashboard |
| `netpulse report --last 24h` | Summary of alerts |
| `netpulse report -e alerts.csv` | Export alerts to CSV |
| `netpulse train` | Manual retrain trigger |
| `netpulse eval` | Show model evaluation metrics |
| `netpulse config` | Edit configuration file |
| `netpulse reset-baseline` | Wipe personal baseline |

## Configuration

Edit `~/.netpulse/config.toml` to customize:

```toml
[general]
interface = ""          # Auto-detect or specify (e.g., "eth0")
mode = "capture"        # "capture" or "alert"
scoring_window = 30     # Seconds between scoring windows

[models]
isolation_forest_contamination = 0.01
autoencoder_bottleneck = 8
autoencoder_epochs = 30
combined_threshold = 1.0

[retraining]
retrain_hour = 2        # 2 AM daily
retrain_window_days = 7
```

## Architecture

NetPulse uses a dual-model approach:

1. **Isolation Forest** - Random partitioning to isolate anomalies
2. **Autoencoder** - Neural network that reconstructs normal traffic; high reconstruction error indicates anomalies

The combined score uses weighted averaging: `0.5 × IF_score + 0.5 × AE_reconstruction_error`

### Feature Engineering

18 features extracted per flow:
- Flow duration, byte counts (log-scaled), packet counts
- Mean/std inter-arrival times (forward and backward)
- TCP flag counts (SYN, ACK, FIN, RST)
- Destination port (normalized)
- Protocol one-hot encoding (TCP/UDP/ICMP)

## Project Structure

```
~/.netpulse/
├── models/
│   ├── current/              # Symlink to latest model
│   └── v2024-01-15/          # Versioned model directory
├── data/
│   └── traffic_log.parquet   # 7-day rolling window
├── alerts.db                 # SQLite alert history
├── reports/                  # ROC curves and metrics
├── eval_history.json         # Training history
└── config.toml               # Configuration
```

## Datasets

For enhanced training, optionally download:

- **CICIDS2018** - https://www.unb.ca/cic/datasets/index.html
- **UNSW-NB15** - https://research.unsw.edu.au/projects/unsw-nb15-dataset

Place in a directory and use `netpulse train --data-path /path/to/data`

## API Usage

```python
from netpulse.capture import PacketCapture
from netpulse.inference import AnomalyDetector
from netpulse.features import FeatureProcessor

# Initialize
capture = PacketCapture(interface="eth0")
detector = AnomalyDetector()
detector.load_models()

# Start capture
capture.start_capture()

# Process flows
flows_df = capture.aggregate_flows()
scored_df, alerts = detector.process_window(flows_df)

for alert in alerts:
    print(f"Alert: {alert['src_ip']} -> {alert['dst_ip']}")
    print(f"  Category: {alert['category']}")
    print(f"  Score: {alert['combined_score']:.3f}")
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please read CONTRIBUTING.md first.

## Support

- Issues: https://github.com/netpulse/netpulse/issues
- Discussions: https://github.com/netpulse/netpulse/discussions
