"""
Simglucose-based blood glucose provider for live simulation.

Uses the simglucose library to generate synthetic CGM data from
FDA-approved UVA/Padova Type 1 Diabetes patient models.
"""

from datetime import datetime, timezone
from typing import Any, Iterator, Literal

import pandas as pd
from loguru import logger
from simglucose.actuator.pump import InsulinPump
from simglucose.controller.basal_bolus_ctrller import (
    CONTROL_QUEST,
    PATIENT_PARA_FILE,
    BBController,
)
from simglucose.controller.base import Action, Controller
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import CustomScenario

from config.settings import settings

# Type alias for insulin modes
InsulinMode = Literal["none", "basal", "basal-bolus"]


def parse_meal_schedule(meal_string: str) -> list[tuple[int, int]]:
    """Parse meal schedule string into list of (hour, carbs) tuples.

    Args:
        meal_string: Comma-separated "hour:carbs" pairs, e.g., "7:45,12:70,18:80"

    Returns:
        List of (hour, carbs) tuples
    """
    meals = []
    for item in meal_string.split(","):
        hour, carbs = item.strip().split(":")
        meals.append((int(hour), int(carbs)))
    return meals


class NoInsulinController(Controller):
    """Controller that delivers no insulin (open-loop)."""

    def __init__(self, init_state: float = 0.0):
        super().__init__(init_state)

    def policy(self, observation, reward, done, **kwargs) -> Action:
        return Action(basal=0, bolus=0)

    def reset(self):
        pass


class BasalOnlyController(Controller):
    """Controller that delivers only basal insulin based on patient parameters."""

    def __init__(self, patient: T1DPatient):
        super().__init__(0)
        # Calculate basal rate from patient's steady-state parameters
        # u2ss is the steady-state insulin rate, BW is body weight
        self._basal_rate = patient._params.u2ss * patient._params.BW / 6000
        logger.debug(f"Basal rate: {self._basal_rate:.4f} U/min")

    def policy(self, observation, reward, done, **kwargs) -> Action:
        return Action(basal=self._basal_rate, bolus=0)

    def reset(self):
        pass


class ConfigurableController(Controller):
    """Insulin controller with user-configurable parameters.

    This controller allows overriding the default patient-specific insulin
    parameters (basal rate, carb ratio, correction factor, target glucose)
    and supports pre-bolus timing for meal boluses.

    When parameters are None, patient-specific defaults from simglucose's
    quest CSV are used, producing identical behavior to BBController.

    Algorithmic Documentation:
    --------------------------

    **Basal Rate:**
    Continuous background insulin to control glucose between meals.
    - Default formula: basal = u2ss × BW / 6000 (U/min)
    - Where u2ss = steady-state insulin (pmol/L/kg), BW = body weight (kg)
    - User override: specified in U/hr, converted to U/min internally

    **Carbohydrate Ratio (CR):**
    Grams of carbohydrates covered by 1 unit of insulin.
    - Meal bolus formula: bolus = carbs_consumed / CR
    - Lower CR = more insulin per carb (more aggressive)
    - Approximation: CR ≈ 500 / TDD (Total Daily Dose)

    **Correction Factor (CF):**
    Blood glucose drop (mg/dL) per 1 unit of insulin.
    - Correction formula: bolus = (current_glucose - target) / CF
    - Only applied when glucose > 150 mg/dL (correction threshold)
    - Approximation: CF ≈ 1800 / TDD

    **Target Glucose:**
    Goal blood glucose for correction boluses.
    - Typical range: 100-150 mg/dL
    - Default: 140 mg/dL (matches BBController)

    **Pre-Bolus Time:**
    Minutes before a meal to deliver the meal bolus.
    - Allows insulin to start acting before glucose rises from food
    - Optimal: 15-30 minutes for rapid-acting insulin
    - Default: 0 (bolus at meal time, matching BBController)

    Clinical Context:
    -----------------
    - Basal typically 40-50% of Total Daily Dose
    - CR varies by time of day (often lower at breakfast due to dawn phenomenon)
    - CF varies by individual insulin sensitivity
    - Pre-bolusing reduces post-meal glucose spikes but requires meal timing precision
    """

    # Validation ranges (clinical safety bounds)
    BASAL_RATE_MIN = 0.0  # U/hr
    BASAL_RATE_MAX = 5.0  # U/hr
    TARGET_GLUCOSE_MIN = 70  # mg/dL
    TARGET_GLUCOSE_MAX = 200  # mg/dL
    CARB_RATIO_MIN = 1  # g/U
    CARB_RATIO_MAX = 50  # g/U
    CORRECTION_FACTOR_MIN = 5  # mg/dL/U
    CORRECTION_FACTOR_MAX = 200  # mg/dL/U
    PRE_BOLUS_MIN = 0  # minutes
    PRE_BOLUS_MAX = 45  # minutes

    # Correction threshold - only correct when glucose exceeds this
    CORRECTION_THRESHOLD = 150  # mg/dL (matches BBController)

    def __init__(
        self,
        patient: T1DPatient,
        meal_schedule: list[tuple[int, int]],
        basal_rate: float | None = None,
        target_glucose: float = 140.0,
        carb_ratio: float | None = None,
        correction_factor: float | None = None,
        pre_bolus_minutes: int = 0,
    ):
        """Initialize the configurable insulin controller.

        Args:
            patient: T1DPatient instance for default parameter lookup.
            meal_schedule: List of (hour, carbs) tuples for pre-bolus lookahead.
            basal_rate: Basal insulin rate in U/hr. None = use patient default.
            target_glucose: Target blood glucose in mg/dL for corrections.
            carb_ratio: Carbohydrate ratio in g/U. None = use patient default.
            correction_factor: Correction factor in mg/dL/U. None = use patient default.
            pre_bolus_minutes: Minutes before meal to deliver bolus. 0 = at meal time.

        Raises:
            ValueError: If any parameter is outside valid clinical range.
        """
        super().__init__(0)

        self._patient = patient
        self._patient_name = patient.name
        self._meal_schedule = meal_schedule
        self._pre_bolus_minutes = pre_bolus_minutes
        self._target_glucose = target_glucose

        # Load patient defaults from simglucose CSV files
        self._quest = pd.read_csv(CONTROL_QUEST)
        self._patient_params = pd.read_csv(PATIENT_PARA_FILE)

        # Get patient-specific parameters
        if any(self._quest.Name.str.match(self._patient_name)):
            quest_row = self._quest[self._quest.Name.str.match(self._patient_name)]
            params_row = self._patient_params[
                self._patient_params.Name.str.match(self._patient_name)
            ]
            self._default_cr = quest_row.CR.values.item()
            self._default_cf = quest_row.CF.values.item()
            self._u2ss = params_row.u2ss.values.item()
            self._bw = params_row.BW.values.item()
        else:
            # Fallback defaults (same as BBController)
            self._default_cr = 1 / 15  # 15 g/U inverted
            self._default_cf = 1 / 50  # 50 mg/dL/U inverted
            self._u2ss = 1.43
            self._bw = 57.0
            logger.warning(
                f"Patient {self._patient_name} not found in quest CSV, using defaults"
            )

        # Calculate default basal rate: u2ss × BW / 6000 (U/min)
        self._default_basal_rate = self._u2ss * self._bw / 6000

        # Apply overrides or use defaults
        if basal_rate is not None:
            self._validate_range(
                "basal_rate", basal_rate, self.BASAL_RATE_MIN, self.BASAL_RATE_MAX
            )
            self._basal_rate = basal_rate / 60  # Convert U/hr to U/min
            logger.info(f"Basal rate override: {basal_rate:.2f} U/hr")
        else:
            self._basal_rate = self._default_basal_rate
            logger.debug(
                f"Basal rate (patient default): {self._basal_rate * 60:.2f} U/hr"
            )

        self._validate_range(
            "target_glucose",
            target_glucose,
            self.TARGET_GLUCOSE_MIN,
            self.TARGET_GLUCOSE_MAX,
        )

        if carb_ratio is not None:
            self._validate_range(
                "carb_ratio", carb_ratio, self.CARB_RATIO_MIN, self.CARB_RATIO_MAX
            )
            self._carb_ratio = carb_ratio
            logger.info(f"Carb ratio override: {carb_ratio:.1f} g/U")
        else:
            self._carb_ratio = self._default_cr
            logger.debug(f"Carb ratio (patient default): {self._carb_ratio:.1f} g/U")

        if correction_factor is not None:
            self._validate_range(
                "correction_factor",
                correction_factor,
                self.CORRECTION_FACTOR_MIN,
                self.CORRECTION_FACTOR_MAX,
            )
            self._correction_factor = correction_factor
            logger.info(f"Correction factor override: {correction_factor:.1f} mg/dL/U")
        else:
            self._correction_factor = self._default_cf
            logger.debug(
                f"Correction factor (patient default): {self._correction_factor:.1f} mg/dL/U"
            )

        self._validate_range(
            "pre_bolus_minutes",
            pre_bolus_minutes,
            self.PRE_BOLUS_MIN,
            self.PRE_BOLUS_MAX,
        )

        # Track which meals have been pre-bolused (to avoid double-dosing)
        self._pre_bolused_meals: set[int] = set()

        # Log configuration summary
        logger.info(
            f"ConfigurableController initialized: "
            f"basal={self._basal_rate * 60:.2f} U/hr, "
            f"target={self._target_glucose} mg/dL, "
            f"CR={self._carb_ratio:.1f} g/U, "
            f"CF={self._correction_factor:.1f} mg/dL/U, "
            f"pre-bolus={self._pre_bolus_minutes} min"
        )

    def _validate_range(
        self, name: str, value: float, min_val: float, max_val: float
    ) -> None:
        """Validate parameter is within acceptable range.

        Args:
            name: Parameter name for error message.
            value: Value to validate.
            min_val: Minimum allowed value.
            max_val: Maximum allowed value.

        Raises:
            ValueError: If value is outside range.
        """
        if value < min_val or value > max_val:
            raise ValueError(
                f"{name} must be between {min_val} and {max_val}, got {value}"
            )

    def _get_simulation_hour(self, info: dict) -> int | None:
        """Extract current simulation hour from step info."""
        sim_time = info.get("time")
        if sim_time is None:
            return None
        if isinstance(sim_time, datetime):
            return sim_time.hour
        # Try parsing if it's a string
        try:
            return datetime.fromisoformat(str(sim_time)).hour
        except (ValueError, TypeError):
            return None

    def _check_pre_bolus(self, info: dict, sample_time: int) -> float:
        """Check if a pre-bolus should be delivered for an upcoming meal.

        Args:
            info: Step info dict containing simulation time.
            sample_time: Simulation sample time in minutes.

        Returns:
            Bolus amount in U/min if pre-bolus needed, 0 otherwise.
        """
        if self._pre_bolus_minutes == 0:
            return 0.0

        sim_time = info.get("time")
        if sim_time is None:
            return 0.0

        if isinstance(sim_time, str):
            try:
                sim_time = datetime.fromisoformat(sim_time)
            except ValueError:
                return 0.0

        current_minutes = sim_time.hour * 60 + sim_time.minute

        for meal_hour, meal_carbs in self._meal_schedule:
            meal_minutes = meal_hour * 60
            pre_bolus_time = meal_minutes - self._pre_bolus_minutes

            # Check if we're at the pre-bolus time (within sample window)
            if (
                meal_hour not in self._pre_bolused_meals
                and pre_bolus_time <= current_minutes < pre_bolus_time + sample_time
            ):
                # Calculate meal bolus
                bolus = meal_carbs / self._carb_ratio
                self._pre_bolused_meals.add(meal_hour)
                logger.info(
                    f"Pre-bolus for {meal_hour:02d}:00 meal ({meal_carbs}g): "
                    f"{bolus:.2f}U delivered at {sim_time.strftime('%H:%M')}"
                )
                return bolus / sample_time  # Convert to U/min

        return 0.0

    def policy(self, observation, reward, done, **kwargs) -> Action:
        """Compute insulin action based on current state.

        This method replicates BBController logic with configurable parameters:
        1. Basal insulin is delivered continuously
        2. Meal bolus is calculated when meal > 0 (or via pre-bolus)
        3. Correction bolus is added when meal > 0 AND glucose > 150 mg/dL
           (matching BBController behavior - no standalone corrections)

        Args:
            observation: Current observation with CGM reading.
            reward: Reward from environment (unused).
            done: Whether episode is done (unused).
            **kwargs: Additional info including meal, patient_name, sample_time.

        Returns:
            Action with basal and bolus rates in U/min.
        """
        sample_time = kwargs.get("sample_time", 3)
        meal = kwargs.get("meal", 0)  # g/min
        glucose = observation.CGM if observation is not None else 140.0

        basal = self._basal_rate
        bolus = 0.0

        # Check for pre-bolus delivery
        pre_bolus = self._check_pre_bolus(kwargs, sample_time)
        if pre_bolus > 0:
            bolus += pre_bolus

        # Calculate meal bolus + correction (only when meal present)
        # This matches BBController behavior where correction is only applied at meal time
        if meal > 0:
            meal_hour = self._get_simulation_hour(kwargs)

            # Only calculate meal bolus if not already pre-bolused
            if meal_hour is None or meal_hour not in self._pre_bolused_meals:
                meal_carbs = meal * sample_time  # Convert g/min to total g
                meal_bolus = meal_carbs / self._carb_ratio

                # Add correction if glucose above threshold (BBController behavior)
                correction = 0.0
                if glucose > self.CORRECTION_THRESHOLD:
                    correction = (
                        glucose - self._target_glucose
                    ) / self._correction_factor

                total_bolus = meal_bolus + correction

                logger.debug(
                    f"Meal bolus: {meal_carbs:.0f}g / {self._carb_ratio:.1f} CR = {meal_bolus:.2f}U"
                )
                if correction > 0:
                    logger.debug(
                        f"Correction: ({glucose:.0f} - {self._target_glucose}) / "
                        f"{self._correction_factor:.1f} CF = {correction:.2f}U"
                    )

                bolus += total_bolus / sample_time  # Convert to U/min

        return Action(basal=basal, bolus=bolus)

    def reset(self):
        """Reset controller state for new simulation."""
        self._pre_bolused_meals.clear()


class SimglucoseProvider:
    """Provider for simglucose-based synthetic CGM data.

    Generates realistic glucose readings using the UVA/Padova T1D model
    with configurable meal scenarios, virtual patients, and insulin modes.
    """

    def __init__(
        self,
        patient_name: str | None = None,
        meal_schedule: list[tuple[int, int]] | None = None,
        seed: int | None = None,
        insulin_mode: InsulinMode | None = None,
        basal_rate: float | None = None,
        target_glucose: float | None = None,
        carb_ratio: float | None = None,
        correction_factor: float | None = None,
        pre_bolus_minutes: int | None = None,
    ):
        """Initialize the simglucose provider.

        Args:
            patient_name: Virtual patient name (e.g., "adult#001", "adolescent#001").
                         Defaults to settings.SIMGLUCOSE_PATIENT.
            meal_schedule: List of (hour, carbs) tuples for meal times.
                          Defaults to settings.SIMGLUCOSE_MEALS parsed.
            seed: Random seed for reproducibility. Defaults to settings.SIMGLUCOSE_SEED.
            insulin_mode: Insulin delivery mode - "none", "basal", or "basal-bolus".
                         Defaults to settings.SIMGLUCOSE_INSULIN_MODE ("basal").
            basal_rate: Basal insulin rate override in U/hr. None = patient default.
            target_glucose: Target glucose for corrections in mg/dL. None = 140 mg/dL.
            carb_ratio: Carbohydrate ratio override in g/U. None = patient default.
            correction_factor: Correction factor override in mg/dL/U. None = patient default.
            pre_bolus_minutes: Minutes before meal to deliver bolus. None = 0 (at meal time).
        """
        self.patient_name = patient_name or settings.SIMGLUCOSE_PATIENT
        self.seed = seed if seed is not None else settings.SIMGLUCOSE_SEED
        self.insulin_mode = insulin_mode or getattr(
            settings, "SIMGLUCOSE_INSULIN_MODE", "basal"
        )

        # Parse meal schedule
        if meal_schedule is None:
            self.meal_schedule = parse_meal_schedule(settings.SIMGLUCOSE_MEALS)
        else:
            self.meal_schedule = meal_schedule

        # Configurable insulin parameters (None = use patient defaults)
        self.basal_rate = (
            basal_rate
            if basal_rate is not None
            else getattr(settings, "SIMGLUCOSE_BASAL_RATE", None)
        )
        self.target_glucose = (
            target_glucose
            if target_glucose is not None
            else getattr(settings, "SIMGLUCOSE_TARGET_GLUCOSE", 140.0)
        )
        self.carb_ratio = (
            carb_ratio
            if carb_ratio is not None
            else getattr(settings, "SIMGLUCOSE_CARB_RATIO", None)
        )
        self.correction_factor = (
            correction_factor
            if correction_factor is not None
            else getattr(settings, "SIMGLUCOSE_CORRECTION_FACTOR", None)
        )
        self.pre_bolus_minutes = (
            pre_bolus_minutes
            if pre_bolus_minutes is not None
            else getattr(settings, "SIMGLUCOSE_PRE_BOLUS_MINUTES", 0)
        )

        self._env = None
        self._patient = None
        self._controller = None
        self._start_time = None

    def _create_controller(self) -> Controller:
        """Create the appropriate insulin controller based on mode and parameters.

        Uses ConfigurableController when:
        - insulin_mode is "basal-bolus", or
        - Any insulin parameter override is specified (basal_rate, carb_ratio, etc.)

        Uses simpler controllers (NoInsulinController, BasalOnlyController) when
        no parameter overrides are specified and mode is "none" or "basal".
        """
        # Check if any configurable parameters are overridden
        has_overrides = any(
            [
                self.basal_rate is not None,
                self.carb_ratio is not None,
                self.correction_factor is not None,
                self.pre_bolus_minutes > 0,
                self.target_glucose != 140.0,
            ]
        )

        if self.insulin_mode == "none" and not has_overrides:
            logger.info("Insulin mode: none (open-loop)")
            return NoInsulinController()

        elif self.insulin_mode == "basal" and not has_overrides:
            logger.info("Insulin mode: basal only")
            return BasalOnlyController(self._patient)

        elif self.insulin_mode == "basal-bolus" or has_overrides:
            # Use ConfigurableController for basal-bolus or when overrides present
            mode_str = self.insulin_mode
            if has_overrides and self.insulin_mode != "basal-bolus":
                mode_str = f"{self.insulin_mode} (with overrides)"
            logger.info(f"Insulin mode: {mode_str} - using ConfigurableController")

            return ConfigurableController(
                patient=self._patient,
                meal_schedule=self.meal_schedule,
                basal_rate=self.basal_rate,
                target_glucose=self.target_glucose,
                carb_ratio=self.carb_ratio,
                correction_factor=self.correction_factor,
                pre_bolus_minutes=self.pre_bolus_minutes,
            )

        else:
            raise ValueError(f"Unknown insulin mode: {self.insulin_mode}")

    def _create_env(self, start_time: datetime) -> T1DSimEnv:
        """Create a new simglucose environment.

        Args:
            start_time: Simulation start datetime

        Returns:
            Configured T1DSimEnv instance
        """
        self._start_time = start_time

        # Create scenario with meal schedule
        scenario = CustomScenario(
            start_time=start_time,
            scenario=self.meal_schedule,
        )

        # Create patient, sensor, pump
        self._patient = T1DPatient.withName(self.patient_name)
        sensor = CGMSensor.withName("Dexcom", seed=self.seed)
        pump = InsulinPump.withName("Insulet")

        # Create controller
        self._controller = self._create_controller()

        # Create environment
        env = T1DSimEnv(self._patient, sensor, pump, scenario)
        return env

    def _minutes_since_midnight(self, dt: datetime) -> int:
        """Calculate minutes since midnight for a datetime."""
        return dt.hour * 60 + dt.minute

    def _get_action(self, observation, reward, done, **info) -> Action:
        """Get insulin action from controller."""
        if self.insulin_mode == "basal-bolus":
            # BBController needs meal info for bolus calculation
            return self._controller.policy(observation, reward, done, **info)
        else:
            return self._controller.policy(observation, reward, done)

    def fast_forward_to_time(self, target_time: datetime) -> float:
        """Fast-forward simulation to a target time.

        Creates a new environment starting at midnight of the target date,
        then steps through to reach the target time.

        Args:
            target_time: Target datetime to fast-forward to

        Returns:
            Current CGM value after fast-forwarding
        """
        # Start at midnight of the target date
        midnight = target_time.replace(hour=0, minute=0, second=0, microsecond=0)
        self._env = self._create_env(midnight)

        # Calculate how many 3-minute steps to reach target time
        minutes_to_advance = self._minutes_since_midnight(target_time)
        steps = minutes_to_advance // 3  # simglucose uses 3-minute steps

        logger.info(
            f"Fast-forwarding {steps} steps ({minutes_to_advance} min) to {target_time.strftime('%H:%M')}"
        )

        # Reset environment and step through
        step = self._env.reset()
        cgm_value = step.observation.CGM

        for _ in range(steps):
            action = self._get_action(
                step.observation, step.reward, step.done, **step.info
            )
            step = self._env.step(action)
            cgm_value = step.observation.CGM

        logger.info(f"Fast-forward complete. CGM: {cgm_value:.1f} mg/dL")
        return cgm_value

    def simulate_glucose_stream(
        self, sync_to_current_time: bool = True, verbose: bool = False
    ) -> Iterator[dict[str, Any]]:
        """Generate a stream of glucose readings with current timestamps.

        Args:
            sync_to_current_time: If True, fast-forward to current time of day.
                                 If False, start from midnight.
            verbose: If True, log detailed information for each reading.

        Yields:
            Dict with timestamp, time (ISO), value, and patient ID
        """
        now = datetime.now(timezone.utc)

        if sync_to_current_time:
            # Fast-forward to current time
            self.fast_forward_to_time(now)
            # Use a no-op action for first step (controller will get proper obs next iteration)
            step = self._env.step(Action(basal=0, bolus=0))
        else:
            # Start from midnight
            midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
            self._env = self._create_env(midnight)
            step = self._env.reset()

        while True:
            # Get action from controller
            action = self._get_action(
                step.observation, step.reward, step.done, **step.info
            )

            # Step the simulation
            step = self._env.step(action)
            cgm_value = step.observation.CGM

            # Use current system time for the reading
            current_time = datetime.now(timezone.utc)

            values = {
                "timestamp": current_time.timestamp(),
                "time": current_time.isoformat(),
                "value": float(cgm_value),
                "patient": self.patient_name,
            }

            if verbose:
                sim_time = step.info.get("time", "N/A")
                meal = step.info.get("meal", 0)
                meal_str = f" [MEAL: {meal}g]" if meal > 0 else ""
                basal_str = f" | Basal: {action.basal:.4f}" if action.basal > 0 else ""
                bolus_str = f" | Bolus: {action.bolus:.2f}" if action.bolus > 0 else ""
                logger.info(
                    f"CGM: {cgm_value:.1f} mg/dL | Sim: {sim_time}{meal_str}{basal_str}{bolus_str}"
                )

            yield values

    def get_glycose_levels(self, start: int = 0) -> Any:
        """Not implemented for simglucose provider (live simulation only)."""
        raise NotImplementedError(
            "SimglucoseProvider generates live data, not historical levels"
        )

    def tsfresh_dataframe(self, truncate: int = 0) -> pd.DataFrame:
        """Not implemented for simglucose provider (live simulation only)."""
        raise NotImplementedError(
            "SimglucoseProvider generates live data, not batch dataframes"
        )
