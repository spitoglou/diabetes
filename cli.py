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


@serve_app.command("simglucose-client")
def serve_simglucose_client(
    patient: str = typer.Option(
        None,
        "--patient",
        "-p",
        help="Virtual patient name (e.g., adult#001, adolescent#001)",
    ),
    seed: int = typer.Option(
        None, "--seed", "-s", help="Random seed for reproducibility"
    ),
    insulin_mode: str = typer.Option(
        None,
        "--insulin-mode",
        "-i",
        help="Insulin mode: none, basal (default), basal-bolus",
    ),
    basal_rate: float = typer.Option(
        None,
        "--basal-rate",
        "-b",
        help="Basal insulin rate in U/hr (0.0-5.0). Overrides patient default.",
    ),
    target_glucose: float = typer.Option(
        None,
        "--target-glucose",
        "-t",
        help="Target glucose for corrections in mg/dL (70-200). Default: 140.",
    ),
    carb_ratio: float = typer.Option(
        None,
        "--carb-ratio",
        "-c",
        help="Carb ratio in g/U (1-50). Grams of carbs per unit insulin.",
    ),
    correction_factor: float = typer.Option(
        None,
        "--correction-factor",
        "-f",
        help="Correction factor in mg/dL/U (5-200). BG drop per unit insulin.",
    ),
    pre_bolus_minutes: int = typer.Option(
        None,
        "--pre-bolus-minutes",
        "-B",
        help="Minutes before meal to deliver bolus (0-45). Default: 0.",
    ),
    no_sync: bool = typer.Option(
        False, "--no-sync", help="Start from midnight instead of current time"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Start the simglucose CGM simulation client.

    Streams synthetic CGM data using the simglucose library with FDA-approved
    UVA/Padova virtual patient models. By default, fast-forwards to current
    time of day so meal effects are realistic.

    Available patients: adult#001-010, adolescent#001-010, child#001-010

    Insulin modes:
      none        - Open-loop, no insulin (glucose will rise after meals)
      basal       - Basal insulin only (default, maintains baseline)
      basal-bolus - Full basal-bolus controller (handles meal spikes)

    Configurable insulin parameters (override patient defaults):
      --basal-rate       - Continuous background insulin (U/hr)
      --target-glucose   - Goal BG for correction boluses (mg/dL)
      --carb-ratio       - Carbs covered per unit insulin (g/U)
      --correction-factor - BG drop per unit insulin (mg/dL/U)
      --pre-bolus-minutes - Deliver meal bolus early (minutes)

    Examples:
      # Default basal-bolus with patient parameters
      uv run python cli.py serve simglucose-client -i basal-bolus

      # Custom insulin settings
      uv run python cli.py serve simglucose-client -i basal-bolus \\
          --basal-rate 1.2 --target-glucose 120 --carb-ratio 12

      # Pre-bolus 15 minutes before meals
      uv run python cli.py serve simglucose-client -i basal-bolus -B 15
    """
    from config.settings import settings

    patient_name = patient or settings.SIMGLUCOSE_PATIENT
    mode = insulin_mode or settings.SIMGLUCOSE_INSULIN_MODE
    typer.echo(f"Starting simglucose client for patient {patient_name}...")
    typer.echo(f"Insulin mode: {mode}")

    # Show configured parameters
    if basal_rate is not None:
        typer.echo(f"Basal rate override: {basal_rate} U/hr")
    if target_glucose is not None:
        typer.echo(f"Target glucose: {target_glucose} mg/dL")
    if carb_ratio is not None:
        typer.echo(f"Carb ratio override: {carb_ratio} g/U")
    if correction_factor is not None:
        typer.echo(f"Correction factor override: {correction_factor} mg/dL/U")
    if pre_bolus_minutes is not None and pre_bolus_minutes > 0:
        typer.echo(f"Pre-bolus time: {pre_bolus_minutes} minutes")

    if not no_sync:
        typer.echo("Syncing to current time of day...")
    if seed is not None:
        typer.echo(f"Using random seed: {seed}")

    from scripts.serving.client import stream_simglucose_data

    stream_simglucose_data(
        send_to_service=True,
        verbose=verbose,
        patient=patient,
        seed=seed,
        sync=not no_sync,
        insulin_mode=insulin_mode,
        basal_rate=basal_rate,
        target_glucose=target_glucose,
        carb_ratio=carb_ratio,
        correction_factor=correction_factor,
        pre_bolus_minutes=pre_bolus_minutes,
    )


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
