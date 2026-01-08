#!/usr/bin/env python3
"""
Unified CLI for the Diabetes Blood Glucose Prediction System.

Usage:
    uv run python cli.py --help
    uv run python cli.py train --help
    uv run python cli.py serve --help
"""

from typing import Optional

import typer

app = typer.Typer(
    name="diabetes",
    help="Diabetes Blood Glucose Prediction System CLI",
    no_args_is_help=True,
)

# Sub-applications
train_app = typer.Typer(help="Training commands")
serve_app = typer.Typer(help="Server commands")
data_app = typer.Typer(help="Data management commands")

app.add_typer(train_app, name="train")
app.add_typer(serve_app, name="serve")
app.add_typer(data_app, name="data")


@app.command()
def info():
    """Show current configuration and system info."""
    from config.settings import settings

    typer.echo("Diabetes Blood Glucose Prediction System")
    typer.echo("=" * 45)
    typer.echo(f"Patient ID:          {settings.OHIO_ID}")
    typer.echo(f"Window Steps:        {settings.WINDOW_STEPS}")
    typer.echo(f"Prediction Horizon:  {settings.PREDICTION_HORIZON}")
    typer.echo(f"MongoDB Database:    {settings.MONGO_DATABASE}")
    typer.echo(f"Server Port:         {settings.PORT}")
    typer.echo(f"Neptune Enabled:     {settings.neptune_enabled}")


# =============================================================================
# Training Commands
# =============================================================================


@train_app.command("simple")
def train_simple(
    patient: str = typer.Option("559", "--patient", "-p", help="Patient ID"),
    window: int = typer.Option(12, "--window", "-w", help="Window size in steps"),
    horizon: int = typer.Option(
        6, "--horizon", "-h", help="Prediction horizon in steps"
    ),
):
    """Run simple training script for a patient."""
    typer.echo(
        f"Training model for patient {patient} (window={window}, horizon={horizon})"
    )

    from src.helpers.experiment import Experiment

    exp = Experiment(
        patient=patient,
        window=window,
        horizon=horizon,
        speed=3,
        enable_neptune=False,
    )
    exp.run_experiment()
    typer.echo("Training complete!")


@train_app.command("full")
def train_full(
    patient: str = typer.Option("559", "--patient", "-p", help="Patient ID"),
    window: int = typer.Option(12, "--window", "-w", help="Window size in steps"),
    horizon: int = typer.Option(
        6, "--horizon", "-h", help="Prediction horizon in steps"
    ),
    neptune: bool = typer.Option(
        True, "--neptune/--no-neptune", help="Enable Neptune logging"
    ),
    speed: int = typer.Option(
        1, "--speed", "-s", help="Speed setting (1=full, 2=medium, 3=fast)"
    ),
):
    """Run full experiment with all model comparisons."""
    typer.echo(f"Running full experiment for patient {patient}")

    from src.helpers.experiment import Experiment

    exp = Experiment(
        patient=patient,
        window=window,
        horizon=horizon,
        speed=speed,
        enable_neptune=neptune,
    )
    exp.run_experiment()
    typer.echo("Experiment complete!")


# =============================================================================
# Server Commands
# =============================================================================


@serve_app.command("start")
def serve_start(
    host: str = typer.Option(None, "--host", help="Host to bind to"),
    port: int = typer.Option(None, "--port", "-p", help="Port to bind to"),
):
    """Start the FastAPI server."""
    import uvicorn

    from config.settings import settings

    _host = host or settings.HOST
    _port = port or settings.PORT

    typer.echo(f"Starting server on {_host}:{_port}")

    # Import the app from new location
    from scripts.serving.server import app as fastapi_app

    uvicorn.run(fastapi_app, host=_host, port=_port)


@serve_app.command("client")
def serve_client(
    patient: str = typer.Option(
        None, "--patient", "-p", help="Patient ID (default: from settings)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Start the CGM data streaming client."""
    from config.settings import settings

    patient_id = patient or settings.OHIO_ID
    typer.echo(f"Starting CGM data streaming client for patient {patient_id}...")

    from scripts.serving.client import stream_data

    stream_data(send_to_service=True, verbose=verbose, patient=patient)


@serve_app.command("synced-client")
def serve_synced_client(
    patient: str = typer.Option(
        None, "--patient", "-p", help="Patient ID (default: from settings)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Start the CGM data streaming client with real-time timestamps.

    Streams CGM data using current system date/time instead of historical
    timestamps. The sequence starts from the dataset reading closest to the
    current time of day, making simulations appear as real-time data.
    """
    from config.settings import settings

    patient_id = patient or settings.OHIO_ID
    typer.echo(f"Starting synced CGM client for patient {patient_id}...")
    typer.echo("Using current system time for timestamps")

    from scripts.serving.client import stream_synced_data

    stream_synced_data(send_to_service=True, verbose=verbose, patient=patient)


@serve_app.command("predict")
def serve_predict(
    patient: str = typer.Option(
        None, "--patient", "-p", help="Patient ID (default: from settings)"
    ),
    window: int = typer.Option(
        None, "--window", "-w", help="Window size in steps (default: from settings)"
    ),
    horizon: int = typer.Option(
        None,
        "--horizon",
        "-H",
        help="Prediction horizon in steps (default: from settings)",
    ),
):
    """Start the prediction watcher (monitors MongoDB for new data)."""
    from config.settings import settings

    patient_id = patient or settings.OHIO_ID
    window_steps = window or settings.WINDOW_STEPS
    horizon_steps = horizon or settings.PREDICTION_HORIZON

    typer.echo(
        f"Starting prediction watcher for patient {patient_id} (window={window_steps}, horizon={horizon_steps})..."
    )

    from scripts.serving.load_model_and_predict import run_prediction_watcher

    run_prediction_watcher(patient_id, window_steps, horizon_steps)


# =============================================================================
# Data Commands
# =============================================================================


@data_app.command("check")
def data_check():
    """Check available Ohio dataset files."""
    typer.echo("Checking Ohio dataset availability...")

    from scripts.utils.check_datasets import main as check_datasets_main

    check_datasets_main()


@data_app.command("generate")
def data_generate(
    patient: int = typer.Option(559, "--patient", "-p", help="Patient ID"),
    window: int = typer.Option(12, "--window", "-w", help="Window size"),
    horizon: int = typer.Option(6, "--horizon", "-h", help="Prediction horizon"),
):
    """Generate feature dataset for a patient."""
    typer.echo(f"Generating dataset for patient {patient}...")

    from src.helpers.experiment import create_tsfresh_dataframe

    # Generate train dataset
    train_params = {
        "ohio_no": patient,
        "scope": "train",
        "train_ds_size": 0,
        "window_size": window,
        "prediction_horizon": horizon,
        "minimal_features": False,
    }
    create_tsfresh_dataframe(train_params)

    # Generate test dataset
    test_params = {
        "ohio_no": patient,
        "scope": "test",
        "train_ds_size": 0,
        "window_size": window,
        "prediction_horizon": horizon,
        "minimal_features": False,
    }
    create_tsfresh_dataframe(test_params)

    typer.echo("Dataset generation complete!")


@data_app.command("mongo-test")
def data_mongo_test():
    """Test MongoDB connection."""
    typer.echo("Testing MongoDB connection...")

    from src.mongo import MongoDB

    mongo = MongoDB()
    mongo.ping()
    mongo.list_databases()
    typer.echo("MongoDB connection successful!")


if __name__ == "__main__":
    app()
