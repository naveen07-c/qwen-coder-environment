# NetPulse Scripts

This folder contains platform-specific scripts to set up and run NetPulse with a single command.

## Available Scripts

### Linux/Unix (`run.sh`)
```bash
# Make executable (if needed)
chmod +x scripts/run.sh

# Run setup and show help
./scripts/run.sh

# Start monitoring (requires sudo)
sudo ./scripts/run.sh start

# View status dashboard
./scripts/run.sh status

# Generate reports
./scripts/run.sh report --last 24h
```

### Windows (`run.bat`)
```batch
# Run setup and show help
scripts\run.bat

# Start monitoring (requires Administrator)
scripts\run.bat start

# View status dashboard
scripts\run.bat status

# Generate reports
scripts\run.bat report --last 24h
```

### macOS (`run-macos.sh`)
```bash
# Make executable (if needed)
chmod +x scripts/run-macos.sh

# Run setup and show help
./scripts/run-macos.sh

# Start monitoring (requires sudo)
sudo ./scripts/run-macos.sh start

# View status dashboard
./scripts/run-macos.sh status

# Generate reports
./scripts/run-macos.sh report --last 24h
```

## What the Scripts Do

Each script performs the following steps automatically:

1. **Checks Python installation** - Verifies Python 3.8+ is available
2. **Creates virtual environment** - Sets up an isolated `venv` directory
3. **Installs dependencies** - Installs all required packages from `requirements.txt`
4. **Creates directories** - Sets up `~/.netpulse/` structure for models, data, and reports
5. **Runs NetPulse** - Executes the specified command or shows help

## CLI Commands

After running the setup script, you can use any of these commands:

| Command | Description | Requires Sudo |
|---------|-------------|---------------|
| `start` | Begin network monitoring | Yes |
| `start --interface <name>` | Monitor specific interface | Yes |
| `status` | Live anomaly dashboard | No |
| `report --last <24h\|7d>` | Alert summary | No |
| `report --last 7d --export file.csv` | Export alerts to CSV | No |
| `train` | Manual model retraining | No |
| `eval` | Run evaluation metrics | No |
| `config` | Edit configuration file | No |
| `reset-baseline` | Clear personal baseline | No |

## Network Interfaces

### Linux
List interfaces with: `ip link show` or `ifconfig -a`
Common names: `eth0`, `wlan0`, `enp0s3`

### Windows
List interfaces with: `netsh interface show interface`
Common names: `Ethernet`, `Wi-Fi`

### macOS
List interfaces with: `networksetup -listallhardwareports`
Common names: `en0` (Wi-Fi), `en1` (Ethernet)

## Troubleshooting

### Permission Denied
Ensure you're running with `sudo` for monitoring commands:
```bash
sudo ./scripts/run.sh start
```

### Python Not Found
Install Python 3.8 or higher:
- **Ubuntu/Debian**: `sudo apt install python3 python3-venv python3-pip`
- **macOS**: `brew install python@3.11`
- **Windows**: Download from [python.org](https://python.org)

### Virtual Environment Issues
Delete the `venv` folder and re-run the script:
```bash
rm -rf venv
./scripts/run.sh
```
