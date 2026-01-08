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
    data_source: str = typer.Option(
        "ohio",
        "--data-source",
        "-d",
        help="Data source: ohio (real patient) or simglucose (synthetic)",
    ),
    interval: int = typer.Option(
        None,
        "--interval",
        "-i",
        help="CGM sample interval in minutes. Auto-detected if not set (ohio=5, simglucose=3)",
    ),
    simulation_days: int = typer.Option(
        14,
        "--simulation-days",
        help="Days to simulate for simglucose training (default: 14). Ignored for Ohio.",
    ),
):
    """Run simple training script for a patient.

    The sample interval determines prediction time: horizon * interval = minutes ahead.
    For simglucose, the interval is auto-detected as 3 minutes (Dexcom sensor).

    Examples:
      # Ohio data (5-min intervals): 6 steps * 5 min = 30 min prediction
      uv run python cli.py train simple -p 559 -w 12 -h 6

      # Simglucose data (3-min intervals): 6 steps * 3 min = 18 min prediction
      uv run python cli.py train simple -p adult#001 -w 12 -h 6 -d simglucose

      # Simglucose with custom simulation duration
      uv run python cli.py train simple -p adult#001 -d simglucose --simulation-days 21
    """
    from src.bgc_providers.factory import get_sample_interval

    # Auto-detect interval from data source if not specified
    sample_interval = (
        interval if interval is not None else get_sample_interval(data_source)
    )
    prediction_minutes = horizon * sample_interval

    typer.echo(f"Training model for patient {patient}")
    typer.echo(f"  Data source: {data_source}")
    typer.echo(
        f"  Window: {window} steps, Horizon: {horizon} steps, Interval: {sample_interval} min"
    )
    typer.echo(f"  Predicting {prediction_minutes} minutes ahead")
    if data_source == "simglucose":
        typer.echo(f"  Simulation: {simulation_days} days")

    from src.helpers.experiment import Experiment

    exp = Experiment(
        patient=patient,
        window=window,
        horizon=horizon,
        data_source=data_source,
        min_per_measure=interval,  # Pass None to let Experiment auto-detect
        simulation_days=simulation_days,
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
    data_source: str = typer.Option(
        "ohio",
        "--data-source",
        "-d",
        help="Data source: ohio (real patient) or simglucose (synthetic)",
    ),
    interval: int = typer.Option(
        None,
        "--interval",
        "-i",
        help="CGM sample interval in minutes. Auto-detected if not set (ohio=5, simglucose=3)",
    ),
    simulation_days: int = typer.Option(
        14,
        "--simulation-days",
        help="Days to simulate for simglucose training (default: 14). Ignored for Ohio.",
    ),
    neptune: bool = typer.Option(
        True, "--neptune/--no-neptune", help="Enable Neptune logging"
    ),
    speed: int = typer.Option(
        1, "--speed", "-s", help="Speed setting (1=full, 2=medium, 3=fast)"
    ),
):
    """Run full experiment with all model comparisons.

    The sample interval determines prediction time: horizon * interval = minutes ahead.
    For simglucose, the interval is auto-detected as 3 minutes (Dexcom sensor).

    Examples:
      # Ohio data (5-min intervals)
      uv run python cli.py train full -p 559 -w 12 -h 6

      # Simglucose data (3-min intervals)
      uv run python cli.py train full -p adult#001 -w 12 -h 6 -d simglucose --no-neptune

      # Simglucose with custom simulation duration
      uv run python cli.py train full -p adult#001 -d simglucose --simulation-days 21 --no-neptune
    """
    from src.bgc_providers.factory import get_sample_interval

    # Auto-detect interval from data source if not specified
    sample_interval = (
        interval if interval is not None else get_sample_interval(data_source)
    )
    prediction_minutes = horizon * sample_interval

    typer.echo(f"Running full experiment for patient {patient}")
    typer.echo(f"  Data source: {data_source}")
    typer.echo(
        f"  Window: {window} steps, Horizon: {horizon} steps, Interval: {sample_interval} min"
    )
    typer.echo(f"  Predicting {prediction_minutes} minutes ahead")
    if data_source == "simglucose":
        typer.echo(f"  Simulation: {simulation_days} days")

    from src.helpers.experiment import Experiment

    exp = Experiment(
        patient=patient,
        window=window,
        horizon=horizon,
        data_source=data_source,
        min_per_measure=interval,  # Pass None to let Experiment auto-detect
        simulation_days=simulation_days,
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
    interval: int = typer.Option(
        None,
        "--interval",
        "-i",
        help="CGM sample interval in minutes (5 for Ohio, 3 for simglucose)",
    ),
):
    """Start the prediction watcher (monitors MongoDB for new data).

    The prediction time is calculated as: horizon_steps * sample_interval.

    Examples:
      # Ohio data (5-min intervals): 6 steps * 5 min = 30 min ahead
      uv run python cli.py serve predict -p 559 -w 12 -H 6

      # Simglucose data (3-min intervals): 6 steps * 3 min = 18 min ahead
      uv run python cli.py serve predict -p adult#001 -w 12 -H 6 -i 3
    """
    from config.settings import settings

    patient_id = patient or settings.OHIO_ID
    window_steps = window or settings.WINDOW_STEPS
    horizon_steps = horizon or settings.PREDICTION_HORIZON
    sample_interval = interval or settings.SAMPLE_INTERVAL

    prediction_minutes = horizon_steps * sample_interval
    typer.echo(f"Starting prediction watcher for patient {patient_id}")
    typer.echo(
        f"  Window: {window_steps} steps, Horizon: {horizon_steps} steps, Interval: {sample_interval} min"
    )
    typer.echo(f"  Predicting {prediction_minutes} minutes ahead")

    from scripts.serving.load_model_and_predict import run_prediction_watcher

    run_prediction_watcher(patient_id, window_steps, horizon_steps, sample_interval)


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
    patient: str = typer.Option("559", "--patient", "-p", help="Patient ID"),
    window: int = typer.Option(12, "--window", "-w", help="Window size"),
    horizon: int = typer.Option(6, "--horizon", "-h", help="Prediction horizon"),
    data_source: str = typer.Option(
        "ohio",
        "--data-source",
        "-d",
        help="Data source: ohio (real patient) or simglucose (synthetic)",
    ),
    simulation_days: int = typer.Option(
        14,
        "--simulation-days",
        help="Days to simulate for simglucose (default: 14). Ignored for Ohio.",
    ),
):
    """Generate feature dataset for a patient.

    Examples:
      # Ohio data
      uv run python cli.py data generate -p 559

      # Simglucose data
      uv run python cli.py data generate -p adult#001 -d simglucose
    """
    typer.echo(f"Generating dataset for patient {patient} (source: {data_source})...")

    from src.helpers.experiment import create_tsfresh_dataframe

    # Generate train dataset
    train_params = {
        "data_source": data_source,
        "patient": patient,
        "scope": "train",
        "train_ds_size": 0,
        "window_size": window,
        "prediction_horizon": horizon,
        "minimal_features": False,
        "simulation_days": simulation_days if data_source == "simglucose" else 0,
    }
    create_tsfresh_dataframe(train_params)

    # Generate test dataset
    test_params = {
        "data_source": data_source,
        "patient": patient,
        "scope": "test",
        "train_ds_size": 0,
        "window_size": window,
        "prediction_horizon": horizon,
        "minimal_features": False,
        "simulation_days": simulation_days // 2 if data_source == "simglucose" else 0,
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


@data_app.command("generate-historical")
def data_generate_historical(
    patient: str = typer.Option(
        "adult#001",
        "--patient",
        "-p",
        help="Virtual patient name (e.g., adult#001, adolescent#001)",
    ),
    start: str = typer.Option(
        ...,
        "--start",
        "-s",
        help="Start date (YYYY-MM-DD)",
    ),
    end: str = typer.Option(
        ...,
        "--end",
        "-e",
        help="End date (YYYY-MM-DD)",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducibility",
    ),
    insulin_mode: str = typer.Option(
        "basal-bolus",
        "--insulin-mode",
        "-i",
        help="Insulin mode: none, basal, basal-bolus",
    ),
    batch_size: int = typer.Option(
        500,
        "--batch-size",
        "-b",
        help="Log progress every N readings",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Generate data without sending to server",
    ),
):
    """Generate historical simglucose data for a date range.

    Runs a simglucose simulation and generates CGM readings with historical
    timestamps, then sends them to the FastAPI server for storage in MongoDB.

    Uses Dexcom CGM sensor model (3-minute sampling interval).

    Examples:
      # Generate December 2025 data for adult#001
      uv run python cli.py data generate-historical -p adult#001 -s 2025-12-01 -e 2025-12-31

      # Generate one week of data for adolescent#001
      uv run python cli.py data generate-historical -p adolescent#001 -s 2025-12-01 -e 2025-12-07

      # Dry run (don't send to server)
      uv run python cli.py data generate-historical -s 2025-12-01 -e 2025-12-31 --dry-run
    """
    from datetime import datetime, timezone

    # Parse dates
    try:
        start_date = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_date = datetime.strptime(end, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=timezone.utc
        )
    except ValueError as e:
        typer.echo(f"Error parsing dates: {e}", err=True)
        typer.echo("Use format YYYY-MM-DD (e.g., 2025-12-01)", err=True)
        raise typer.Exit(1)

    if end_date < start_date:
        typer.echo("Error: End date must be after start date", err=True)
        raise typer.Exit(1)

    # Calculate expected readings
    total_minutes = (end_date - start_date).total_seconds() / 60
    sample_time = 3  # Dexcom: 3 minutes
    total_readings = int(total_minutes / sample_time)
    days = (end_date - start_date).days + 1

    typer.echo(f"Generating historical simglucose data")
    typer.echo(f"=" * 45)
    typer.echo(f"Patient:        {patient}")
    typer.echo(f"Date range:     {start} to {end} ({days} days)")
    typer.echo(f"Insulin mode:   {insulin_mode}")
    typer.echo(f"Seed:           {seed}")
    typer.echo(f"Total readings: {total_readings:,}")
    typer.echo(f"Send to server: {not dry_run}")
    typer.echo(f"=" * 45)

    from scripts.utils.generate_historical_simglucose import generate_historical_data

    generate_historical_data(
        patient=patient,
        start_date=start_date,
        end_date=end_date,
        seed=seed,
        insulin_mode=insulin_mode,
        send_to_server=not dry_run,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    app()
