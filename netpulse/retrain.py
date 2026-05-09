"""
Nightly retraining scheduler for NetPulse.

Automatically retrains models every night using the past 7 days of traffic,
keeping the baseline fresh as network habits evolve.
"""

import json
import schedule
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from rich.console import Console

from .config import load_config, get_netpulse_dir
from .features import FeatureProcessor, extract_flow_features
from .models import ModelTrainer, save_models, load_models
from .data import save_eval_history

console = Console()


class RetrainingScheduler:
    """
    Handles nightly model retraining.
    
    Runs automatically at configured time, retrains on recent traffic,
    and promotes new models only if they meet quality thresholds.
    """
    
    def __init__(self):
        self.config = load_config()
        self.running = False
        self.last_training: Optional[datetime] = None
        self.training_count = 0
    
    def _should_retrain(self) -> bool:
        """Check if retraining should run based on last training time."""
        if self.last_training is None:
            return True
        
        # Don't retrain more than once per day
        return datetime.now() - self.last_training > timedelta(hours=23)
    
    def _load_recent_traffic(self, days: int = 7) -> pd.DataFrame:
        """Load traffic data from the past N days."""
        traffic_log_path = get_netpulse_dir() / "data" / "traffic_log.parquet"
        
        if not traffic_log_path.exists():
            console.print("[yellow]No traffic log found. Skipping retraining.[/yellow]")
            return pd.DataFrame()
        
        console.print(f"Loading traffic data from {traffic_log_path}...")
        
        try:
            df = pd.read_parquet(traffic_log_path)
        except Exception as e:
            console.print(f"[red]Error loading traffic data: {e}[/red]")
            return pd.DataFrame()
        
        if df.empty:
            console.print("[yellow]Traffic log is empty. Skipping retraining.[/yellow]")
            return df
        
        # Filter to recent traffic (past N days)
        cutoff = time.time() - (days * 86400)
        
        if "start_time" in df.columns:
            df = df[df["start_time"] >= cutoff]
        
        console.print(f"Loaded {len(df)} flows from past {days} days")
        
        return df
    
    def _filter_normal_traffic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out previously flagged anomalous traffic."""
        alerts_db_path = get_netpulse_dir() / "alerts.db"
        
        if not alerts_db_path.exists():
            # No alert history, assume all traffic is normal
            return df
        
        # Load alert timestamps and source IPs
        import sqlite3
        conn = sqlite3.connect(alerts_db_path)
        
        query = """
            SELECT timestamp, src_ip, dst_ip, dst_port 
            FROM alerts 
            WHERE timestamp >= ?
        """
        cutoff = time.time() - (self.config["retraining"]["retrain_window_days"] * 86400)
        
        alerts_df = pd.read_sql_query(query, conn, params=(cutoff,))
        conn.close()
        
        if alerts_df.empty:
            return df
        
        # Remove flows that match alert patterns
        # This is a simplified approach - in production you'd want more sophisticated matching
        for _, alert in alerts_df.iterrows():
            mask = (
                (df["src_ip"] == alert["src_ip"]) &
                (df["dst_ip"] == alert["dst_ip"])
            )
            df = df[~mask]
        
        console.print(f"After filtering anomalies: {len(df)} flows remaining")
        
        return df
    
    def _evaluate_new_model(
        self,
        trainer: ModelTrainer,
        X_val: pd.DataFrame,
    ) -> Dict:
        """Evaluate new model on validation data."""
        # Use reconstruction error as proxy for quality
        # In production with labelled data, would use proper metrics
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainer.autoencoder.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_val).to(device)
            reconstructions = trainer.autoencoder(X_tensor)
            errors = torch.mean((X_tensor - reconstructions) ** 2, dim=1).cpu().numpy()
        
        # Lower reconstruction error on normal data = better model
        metrics = {
            "val_reconstruction_error": float(np.mean(errors)),
            "val_reconstruction_std": float(np.std(errors)),
        }
        
        return metrics
    
    def train(self) -> bool:
        """
        Run the retraining pipeline.
        
        Returns:
            True if new model was promoted, False otherwise
        """
        console.print("\n[bold blue]=== Nightly Retraining ===[/bold blue]\n")
        
        start_time = time.time()
        
        # Load recent traffic
        df = self._load_recent_traffic(
            days=self.config["retraining"]["retrain_window_days"]
        )
        
        if df.empty:
            return False
        
        # Filter out anomalous traffic
        df = self._filter_normal_traffic(df)
        
        if len(df) < 1000:
            console.print("[yellow]Insufficient normal traffic for retraining. Skipping.[/yellow]")
            return False
        
        # Extract features
        console.print("Extracting features...")
        features_df = extract_flow_features(df)
        
        # Prepare data
        feature_processor = FeatureProcessor()
        X_scaled = feature_processor.fit_transform(features_df)
        
        # Split for validation
        from sklearn.model_selection import train_test_split
        X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)
        
        console.print(f"Training set: {len(X_train)}, Validation set: {len(X_val)}")
        
        # Train new models
        trainer = ModelTrainer(
            if_contamination=self.config["models"]["isolation_forest_contamination"],
            ae_bottleneck=self.config["models"]["autoencoder_bottleneck"],
            ae_epochs=self.config["models"]["autoencoder_epochs"],
            ae_batch_size=self.config["models"]["autoencoder_batch_size"],
            ae_learning_rate=self.config["models"]["autoencoder_learning_rate"],
        )
        
        trainer.train_isolation_forest(X_train)
        trainer.train_autoencoder(X_train, X_val)
        
        # Evaluate
        console.print("Evaluating new model...")
        new_metrics = self._evaluate_new_model(trainer, X_val)
        
        # Compare with current model
        current_model_path = Path.home() / ".netpulse" / "models" / "current"
        
        should_promote = True
        
        if current_model_path.exists():
            try:
                _, _, current_meta = load_models(current_model_path)
                current_metrics = current_meta.get("metrics", {})
                
                # Check if new model regresses
                max_regression = self.config["retraining"]["max_f1_regression"]
                
                # For now, just check reconstruction error
                if "val_reconstruction_error" in current_metrics:
                    error_increase = (
                        new_metrics["val_reconstruction_error"] - 
                        current_metrics["val_reconstruction_error"]
                    ) / (current_metrics["val_reconstruction_error"] + 1e-10)
                    
                    if error_increase > max_regression:
                        console.print(
                            f"[yellow]New model shows {error_increase:.2%} regression. "
                            "Keeping old model.[/yellow]"
                        )
                        should_promote = False
            
            except Exception as e:
                console.print(f"[dim]Could not compare with current model: {e}[/dim]")
        
        if should_promote:
            # Save and promote new model
            version = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            # Add evaluation metrics
            all_metrics = {**new_metrics, "combined_f1": 0.0, "combined_auc": 0.0}
            
            save_models(trainer, feature_processor, all_metrics, len(df), version)
            
            # Save eval history
            save_eval_history(all_metrics)
            
            elapsed = time.time() - start_time
            console.print(f"\n[green]✓ Retraining complete in {elapsed:.1f}s[/green]")
            console.print(f"[green]✓ New model promoted: v{version}[/green]")
            
            self.last_training = datetime.now()
            self.training_count += 1
            
            return True
        else:
            console.print("\n[yellow]New model not promoted due to regression[/yellow]")
            return False
    
    def start(self) -> None:
        """Start the background scheduler."""
        if self.running:
            console.print("[yellow]Scheduler already running[/yellow]")
            return
        
        self.running = True
        
        # Schedule daily retraining
        retrain_hour = self.config["retraining"]["retrain_hour"]
        retrain_minute = self.config["retraining"]["retrain_minute"]
        
        schedule.every().day.at(f"{retrain_hour:02d}:{retrain_minute:02d}").do(
            self._scheduled_train
        )
        
        console.print(
            f"[green]✓ Scheduler started - retraining at {retrain_hour:02d}:{retrain_minute:02d} daily[/green]"
        )
        
        # Run scheduler loop
        while self.running:
            schedule.run_pending()
            time.sleep(60)
    
    def _scheduled_train(self) -> None:
        """Wrapper for scheduled training that handles errors."""
        try:
            if self._should_retrain():
                self.train()
            else:
                console.print("[dim]Skipping retraining - already ran today[/dim]")
        except Exception as e:
            console.print(f"[red]Retraining failed: {e}[/red]")
    
    def stop(self) -> None:
        """Stop the scheduler."""
        self.running = False
        console.print("[yellow]Scheduler stopped[/yellow]")
    
    def run_once(self) -> bool:
        """Run retraining immediately (for manual trigger)."""
        return self.train()


import torch
import numpy as np


def main():
    """Entry point for standalone scheduler."""
    console.print("[bold blue]NetPulse Retraining Scheduler[/bold blue]\n")
    
    scheduler = RetrainingScheduler()
    
    try:
        scheduler.start()
    except KeyboardInterrupt:
        console.print("\n[yellow]Scheduler interrupted[/yellow]")
        scheduler.stop()


if __name__ == "__main__":
    main()
