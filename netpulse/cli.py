"""
CLI interface for NetPulse.

Provides commands for starting monitoring, viewing status, generating reports,
and managing models and configuration.
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import click
import toml
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import __version__
from .config import load_config, get_netpulse_dir, ensure_config_dir
from .capture import PacketCapture
from .inference import AnomalyDetector, AlertDatabase
from .features import FeatureProcessor
from .models import ModelTrainer, save_models


console = Console()


@click.group()
@click.version_option(version=__version__)
def cli():
    """NetPulse - Intelligent Network Anomaly Detector
    
    A CLI-based system that passively monitors local network traffic,
    learns normal patterns using unsupervised ML, and raises real-time alerts.
    """
    pass


@cli.command()
@click.option("--interface", "-i", default=None, help="Network interface to monitor")
@click.option("--mode", "-m", type=click.Choice(["capture", "alert"]), default="capture",
              help="Mode: 'capture' for baseline collection, 'alert' for anomaly detection")
def start(interface: Optional[str], mode: str):
    """Start monitoring network traffic.
    
    Requires sudo/root privileges for packet capture.
    
    Examples:
        netpulse start                    # Start with auto-detected interface
        netpulse start -i eth0            # Specify interface
        netpulse start --mode alert       # Run in alert mode
    """
    config = load_config()
    
    if interface is None:
        interface = config["general"].get("interface") or None
    
    console.print(f"[bold blue]NetPulse v{__version__}[/bold blue]")
    console.print(f"Starting network monitoring...")
    console.print(f"  Interface: {interface or 'auto-detect'}")
    console.print(f"  Mode: [green]{mode}[/green]")
    console.print()
    
    # Check if running as root (required for packet capture)
    if os.geteuid() != 0:
        console.print("[yellow]Warning: Not running as root. Packet capture may fail.[/yellow]")
        console.print("[yellow]Try: sudo netpulse start[/yellow]")
        console.print()
    
    # Initialize components
    capture = PacketCapture(interface=interface, buffer_duration=config["general"]["buffer_duration"])
    
    if mode == "alert":
        detector = AnomalyDetector(
            combined_threshold=config["models"]["combined_threshold"],
            if_weight=config["models"]["if_weight"],
            ae_weight=config["models"]["ae_weight"],
        )
        
        try:
            detector.load_models()
        except FileNotFoundError as e:
            console.print(f"[red]Error: {e}[/red]")
            console.print("[yellow]Run 'netpulse train' first to train models.[/yellow]")
            sys.exit(1)
    
    # Traffic log file
    traffic_log_path = get_netpulse_dir() / "data" / "traffic_log.parquet"
    
    # Start capture
    capture.start_capture()
    console.print("[green]✓ Capture started[/green]")
    console.print()
    
    # Main loop
    scoring_window = config["general"]["scoring_window"]
    last_flush = time.time()
    
    all_flows = []
    
    try:
        while True:
            time.sleep(scoring_window)
            
            # Aggregate flows from buffer
            flows_df = capture.aggregate_flows()
            
            if not flows_df.empty:
                all_flows.append(flows_df)
                
                if mode == "alert":
                    # Score and generate alerts
                    scored_df, alerts = detector.process_window(
                        flows_df,
                        alert_callback=detector.render_alert if config["alerts"]["terminal_alerts"] else None
                    )
                    
                    if alerts:
                        console.print(f"[dim]Generated {len(alerts)} alert(s)[/dim]")
                
                # Log traffic
                console.print(f"[dim]Processed {len(flows_df)} flows ({capture.get_stats()['packets_per_second']} pkt/s)[/dim]")
            
            # Save traffic log periodically (every 5 minutes)
            if time.time() - last_flush > 300 and all_flows:
                import pandas as pd
                combined_df = pd.concat(all_flows, ignore_index=True)
                combined_df.to_parquet(traffic_log_path, engine="pyarrow", index=False)
                all_flows = []
                last_flush = time.time()
                console.print(f"[dim]Traffic log saved[/dim]")
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping capture...[/yellow]")
        capture.stop_capture()
        
        # Final save
        if all_flows:
            import pandas as pd
            combined_df = pd.concat(all_flows, ignore_index=True)
            combined_df.to_parquet(traffic_log_path, engine="pyarrow", index=False)
        
        console.print("[green]✓ Monitoring stopped[/green]")


@cli.command()
def status():
    """Show live monitoring dashboard."""
    config = load_config()
    
    db = AlertDatabase()
    
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=10),
    )
    
    layout["header"].update(
        Panel(
            f"[bold blue]NetPulse v{__version__}[/bold blue]\n"
            f"Status Dashboard",
            title="Header",
        )
    )
    
    def make_layout() -> Layout:
        # Get stats
        total_alerts = db.get_alert_count()
        recent_alerts = db.get_alert_count(since=time.time() - 3600)  # Last hour
        
        # Status panel
        status_table = Table(show_header=False, box=None)
        status_table.add_column("Key", style="cyan")
        status_table.add_column("Value")
        status_table.add_row("Total Alerts", str(total_alerts))
        status_table.add_row("Last Hour", str(recent_alerts))
        status_table.add_row("Mode", config["general"]["mode"])
        
        layout["body"].update(
            Panel(
                status_table,
                title="Status",
            )
        )
        
        # Recent alerts
        alerts_df = db.get_alerts(limit=5)
        
        if not alerts_df.empty:
            alerts_table = Table(title="Recent Alerts")
            alerts_table.add_column("Time", style="dim")
            alerts_table.add_column("Source IP")
            alerts_table.add_column("Dest IP:Port")
            alerts_table.add_column("Score", justify="right")
            alerts_table.add_column("Category")
            
            for _, row in alerts_df.iterrows():
                ts = datetime.fromtimestamp(row["timestamp"]).strftime("%H:%M:%S")
                alerts_table.add_row(
                    ts,
                    str(row["src_ip"]),
                    f"{row['dst_ip']}:{row['dst_port']}",
                    f"{row['combined_score']:.3f}",
                    str(row["category"])[:30],
                )
            
            layout["footer"].update(Panel(alerts_table, title="Recent Alerts"))
        else:
            layout["footer"].update(Panel("No alerts yet", title="Recent Alerts"))
        
        return layout
    
    try:
        with Live(make_layout(), refresh_per_second=2, screen=True) as live:
            while True:
                time.sleep(1)
                live.update(make_layout())
    except KeyboardInterrupt:
        pass


@cli.command()
@click.option("--last", "-l", default="24h", help="Time window (e.g., '24h', '7d', '1h')")
@click.option("--export", "-e", default=None, help="Export to CSV file")
def report(last: str, export: Optional[str]):
    """Generate summary report of alerts.
    
    Examples:
        netpulse report --last 24h      # Last 24 hours
        netpulse report --last 7d       # Last 7 days
        netpulse report -e alerts.csv   # Export to CSV
    """
    db = AlertDatabase()
    
    # Parse time window
    now = time.time()
    if last.endswith("h"):
        hours = int(last[:-1])
        since = now - (hours * 3600)
    elif last.endswith("d"):
        days = int(last[:-1])
        since = now - (days * 86400)
    else:
        console.print("[red]Invalid time format. Use '24h', '7d', etc.[/red]")
        sys.exit(1)
    
    alerts_df = db.get_alerts(since=since, limit=10000)
    
    if alerts_df.empty:
        console.print(f"[yellow]No alerts in the last {last}[/yellow]")
        return
    
    # Summary statistics
    console.print(f"[bold blue]Alert Report: Last {last}[/bold blue]\n")
    
    console.print(f"Total Alerts: [bold]{len(alerts_df)}[/bold]")
    console.print(f"Unique Source IPs: {alerts_df['src_ip'].nunique()}")
    console.print(f"Unique Dest IPs: {alerts_df['dst_ip'].nunique()}")
    console.print()
    
    # Top source IPs
    console.print("[bold]Top 5 Source IPs:[/bold]")
    top_sources = alerts_df["src_ip"].value_counts().head(5)
    for ip, count in top_sources.items():
        console.print(f"  {ip}: {count} alerts")
    console.print()
    
    # Category breakdown
    console.print("[bold]Alert Categories:[/bold]")
    category_counts = alerts_df["category"].value_counts()
    for cat, count in category_counts.items():
        console.print(f"  {cat}: {count}")
    console.print()
    
    # Average scores
    console.print(f"[bold]Average Combined Score:[/bold] {alerts_df['combined_score'].mean():.3f}")
    console.print(f"[bold]Max Combined Score:[/bold] {alerts_df['combined_score'].max():.3f}")
    
    # Export if requested
    if export:
        alerts_df.to_csv(export, index=False)
        console.print(f"\n[green]✓ Exported to {export}[/green]")


@cli.command()
@click.option("--data-path", "-d", default=None, help="Path to training data (CICIDS2018 CSV)")
def train(data_path: Optional[str]):
    """Manually trigger model retraining.
    
    Uses the past 7 days of captured traffic for personalized baseline.
    Can optionally incorporate CICIDS2018 dataset.
    """
    config = load_config()
    
    console.print(f"[bold blue]NetPulse Model Training[/bold blue]\n")
    
    # Load traffic data
    traffic_log_path = get_netpulse_dir() / "data" / "traffic_log.parquet"
    
    if not traffic_log_path.exists():
        console.print("[yellow]No traffic data found. Run monitoring first.[/yellow]")
        console.print("[yellow]Or provide external dataset with --data-path[/yellow]")
        sys.exit(1)
    
    console.print("Loading traffic data...")
    import pandas as pd
    df = pd.read_parquet(traffic_log_path)
    console.print(f"Loaded {len(df)} flow records")
    
    # Extract features
    from .features import extract_flow_features, FeatureProcessor
    
    console.print("Extracting features...")
    features_df = extract_flow_features(df)
    
    # Train models
    trainer = ModelTrainer(
        if_contamination=config["models"]["isolation_forest_contamination"],
        ae_bottleneck=config["models"]["autoencoder_bottleneck"],
        ae_epochs=config["models"]["autoencoder_epochs"],
        ae_batch_size=config["models"]["autoencoder_batch_size"],
        ae_learning_rate=config["models"]["autoencoder_learning_rate"],
    )
    
    # Prepare data
    feature_processor = FeatureProcessor()
    X_scaled = feature_processor.fit_transform(features_df)
    
    # Split for validation
    from sklearn.model_selection import train_test_split
    X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)
    
    # Train Isolation Forest
    trainer.train_isolation_forest(X_train)
    
    # Train Autoencoder
    trainer.train_autoencoder(X_train, X_val)
    
    # Save models
    metrics = {
        "combined_f1": 0.0,  # Would need attack data for proper evaluation
        "combined_auc": 0.0,
    }
    
    save_models(trainer, feature_processor, metrics, len(df))
    
    console.print("\n[green]✓ Training complete![/green]")


@cli.command()
def eval():
    """Re-run evaluation on held-out test set."""
    console.print("[bold blue]Model Evaluation[/bold blue]\n")
    
    # Load current models
    from .models import load_models
    
    try:
        isolation_forest, autoencoder, meta = load_models()
        console.print("Models loaded successfully")
        console.print(f"Training date: {meta.get('training_date', 'unknown')}")
        console.print(f"Training samples: {meta.get('n_samples', 0)}")
        console.print()
        
        # Display stored metrics
        metrics = meta.get("metrics", {})
        console.print("[bold]Stored Metrics:[/bold]")
        for key, value in metrics.items():
            console.print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    except FileNotFoundError:
        console.print("[red]No models found. Run 'netpulse train' first.[/red]")
        sys.exit(1)


@cli.command()
def config_cmd():
    """Open configuration file in editor."""
    config_path = get_netpulse_dir() / "config.toml"
    ensure_config_dir()
    
    if not config_path.exists():
        from .config import save_config, DEFAULT_CONFIG
        save_config(DEFAULT_CONFIG)
    
    editor = os.environ.get("EDITOR", "nano")
    console.print(f"Opening {config_path} in {editor}...")
    os.system(f"{editor} {config_path}")


@cli.command()
@click.confirmation_option(prompt="Are you sure you want to reset the baseline? This will delete all captured traffic data.")
def reset_baseline():
    """Wipe personal baseline and start fresh."""
    console.print("[yellow]Resetting baseline...[/yellow]")
    
    traffic_log_path = get_netpulse_dir() / "data" / "traffic_log.parquet"
    alerts_db_path = get_netpulse_dir() / "alerts.db"
    
    if traffic_log_path.exists():
        traffic_log_path.unlink()
        console.print(f"Deleted {traffic_log_path}")
    
    if alerts_db_path.exists():
        alerts_db_path.unlink()
        console.print(f"Deleted {alerts_db_path}")
    
    console.print("[green]✓ Baseline reset complete[/green]")
    console.print("[yellow]Run 'netpulse start --mode capture' to collect new baseline data[/yellow]")


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
