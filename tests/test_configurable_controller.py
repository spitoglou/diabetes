"""
Tests for ConfigurableController equivalence with BBController.

This module verifies that ConfigurableController with default parameters
produces identical insulin actions to simglucose's BBController.
"""

from dataclasses import dataclass
from typing import Any

import pytest

# Skip all tests if simglucose is not available
pytest.importorskip("simglucose")


@dataclass
class MockObservation:
    """Mock observation with CGM reading."""

    CGM: float


class TestConfigurableControllerEquivalence:
    """Test that ConfigurableController matches BBController behavior."""

    @pytest.fixture
    def patient(self):
        """Create a test patient."""
        from simglucose.patient.t1dpatient import T1DPatient

        return T1DPatient.withName("adult#001")

    @pytest.fixture
    def meal_schedule(self):
        """Default meal schedule for testing."""
        return [(7, 45), (12, 70), (16, 15), (18, 80), (23, 10)]

    @pytest.fixture
    def bb_controller(self):
        """Create BBController instance."""
        from simglucose.controller.basal_bolus_ctrller import BBController

        return BBController(target=140)

    @pytest.fixture
    def configurable_controller(self, patient, meal_schedule):
        """Create ConfigurableController with defaults."""
        from src.bgc_providers.simglucose_provider import ConfigurableController

        return ConfigurableController(
            patient=patient,
            meal_schedule=meal_schedule,
            # All defaults - no overrides
        )

    def test_basal_rate_equivalence(self, bb_controller, configurable_controller):
        """Verify basal rates match between controllers."""
        obs = MockObservation(CGM=120.0)
        info = {"patient_name": "adult#001", "meal": 0, "sample_time": 3}

        bb_action = bb_controller.policy(obs, 0, False, **info)
        cfg_action = configurable_controller.policy(obs, 0, False, **info)

        assert abs(bb_action.basal - cfg_action.basal) < 1e-6, (
            f"Basal rate mismatch: BB={bb_action.basal:.6f}, Cfg={cfg_action.basal:.6f}"
        )

    def test_no_meal_no_correction(self, bb_controller, configurable_controller):
        """Verify no bolus when glucose normal and no meal."""
        obs = MockObservation(CGM=120.0)
        info = {"patient_name": "adult#001", "meal": 0, "sample_time": 3}

        bb_action = bb_controller.policy(obs, 0, False, **info)
        cfg_action = configurable_controller.policy(obs, 0, False, **info)

        # Both should have no bolus
        assert bb_action.bolus == 0, (
            f"BBController bolus should be 0, got {bb_action.bolus}"
        )
        assert cfg_action.bolus == 0, (
            f"ConfigurableController bolus should be 0, got {cfg_action.bolus}"
        )

    def test_meal_bolus_equivalence(self, bb_controller, configurable_controller):
        """Verify meal bolus calculations match."""
        obs = MockObservation(CGM=120.0)  # Normal glucose, no correction
        # 15 g/min for 3 min sample = 45g total
        info = {"patient_name": "adult#001", "meal": 15, "sample_time": 3}

        bb_action = bb_controller.policy(obs, 0, False, **info)
        cfg_action = configurable_controller.policy(obs, 0, False, **info)

        assert abs(bb_action.bolus - cfg_action.bolus) < 1e-6, (
            f"Meal bolus mismatch: BB={bb_action.bolus:.6f}, Cfg={cfg_action.bolus:.6f}"
        )

    def test_no_standalone_correction(self, bb_controller, configurable_controller):
        """Verify no bolus when glucose high but no meal (BBController behavior).

        BBController only applies correction boluses at meal times, not standalone.
        This matches real insulin pump behavior where corrections are typically
        combined with meal boluses.
        """
        obs = MockObservation(CGM=200.0)  # High glucose
        info = {"patient_name": "adult#001", "meal": 0, "sample_time": 3}

        bb_action = bb_controller.policy(obs, 0, False, **info)
        cfg_action = configurable_controller.policy(obs, 0, False, **info)

        # Both should have NO bolus (correction only at meal time)
        assert bb_action.bolus == 0, (
            f"BBController should have no standalone correction, got {bb_action.bolus}"
        )
        assert cfg_action.bolus == 0, (
            f"ConfigurableController should have no standalone correction, got {cfg_action.bolus}"
        )

    def test_meal_plus_correction_equivalence(
        self, bb_controller, configurable_controller
    ):
        """Verify meal + correction bolus calculations match."""
        obs = MockObservation(CGM=200.0)  # High glucose
        info = {"patient_name": "adult#001", "meal": 15, "sample_time": 3}  # 45g meal

        bb_action = bb_controller.policy(obs, 0, False, **info)
        cfg_action = configurable_controller.policy(obs, 0, False, **info)

        assert abs(bb_action.bolus - cfg_action.bolus) < 1e-6, (
            f"Meal+correction bolus mismatch: BB={bb_action.bolus:.6f}, "
            f"Cfg={cfg_action.bolus:.6f}"
        )

    @pytest.mark.parametrize(
        "glucose,meal",
        [
            (100.0, 0),  # Low normal, no meal
            (140.0, 0),  # Target, no meal
            (150.0, 0),  # At threshold, no meal
            # Note: BBController gives no bolus when meal=0, even with high glucose
            # so we test high glucose scenarios with meals
            (100.0, 10),  # Normal + small meal (30g)
            (100.0, 20),  # Normal + medium meal (60g)
            (100.0, 30),  # Normal + large meal (90g)
            (151.0, 15),  # Just above threshold + meal
            (200.0, 20),  # High + medium meal
            (250.0, 25),  # Very high + large meal
        ],
    )
    def test_multiple_scenarios(
        self, bb_controller, configurable_controller, glucose, meal
    ):
        """Verify equivalence across multiple glucose/meal scenarios."""
        obs = MockObservation(CGM=glucose)
        info = {"patient_name": "adult#001", "meal": meal, "sample_time": 3}

        bb_action = bb_controller.policy(obs, 0, False, **info)
        cfg_action = configurable_controller.policy(obs, 0, False, **info)

        assert abs(bb_action.basal - cfg_action.basal) < 1e-6, (
            f"Basal mismatch at glucose={glucose}, meal={meal}: "
            f"BB={bb_action.basal:.6f}, Cfg={cfg_action.basal:.6f}"
        )
        assert abs(bb_action.bolus - cfg_action.bolus) < 1e-6, (
            f"Bolus mismatch at glucose={glucose}, meal={meal}: "
            f"BB={bb_action.bolus:.6f}, Cfg={cfg_action.bolus:.6f}"
        )


class TestConfigurableControllerValidation:
    """Test parameter validation in ConfigurableController."""

    @pytest.fixture
    def patient(self):
        """Create a test patient."""
        from simglucose.patient.t1dpatient import T1DPatient

        return T1DPatient.withName("adult#001")

    @pytest.fixture
    def meal_schedule(self):
        """Default meal schedule for testing."""
        return [(7, 45), (12, 70)]

    def test_basal_rate_too_high(self, patient, meal_schedule):
        """Verify error when basal rate exceeds maximum."""
        from src.bgc_providers.simglucose_provider import ConfigurableController

        with pytest.raises(ValueError, match="basal_rate must be between"):
            ConfigurableController(
                patient=patient, meal_schedule=meal_schedule, basal_rate=10.0
            )

    def test_target_glucose_too_low(self, patient, meal_schedule):
        """Verify error when target glucose is below minimum."""
        from src.bgc_providers.simglucose_provider import ConfigurableController

        with pytest.raises(ValueError, match="target_glucose must be between"):
            ConfigurableController(
                patient=patient, meal_schedule=meal_schedule, target_glucose=50.0
            )

    def test_target_glucose_too_high(self, patient, meal_schedule):
        """Verify error when target glucose exceeds maximum."""
        from src.bgc_providers.simglucose_provider import ConfigurableController

        with pytest.raises(ValueError, match="target_glucose must be between"):
            ConfigurableController(
                patient=patient, meal_schedule=meal_schedule, target_glucose=250.0
            )

    def test_carb_ratio_too_low(self, patient, meal_schedule):
        """Verify error when carb ratio is below minimum."""
        from src.bgc_providers.simglucose_provider import ConfigurableController

        with pytest.raises(ValueError, match="carb_ratio must be between"):
            ConfigurableController(
                patient=patient, meal_schedule=meal_schedule, carb_ratio=0.5
            )

    def test_carb_ratio_too_high(self, patient, meal_schedule):
        """Verify error when carb ratio exceeds maximum."""
        from src.bgc_providers.simglucose_provider import ConfigurableController

        with pytest.raises(ValueError, match="carb_ratio must be between"):
            ConfigurableController(
                patient=patient, meal_schedule=meal_schedule, carb_ratio=100.0
            )

    def test_correction_factor_too_low(self, patient, meal_schedule):
        """Verify error when correction factor is below minimum."""
        from src.bgc_providers.simglucose_provider import ConfigurableController

        with pytest.raises(ValueError, match="correction_factor must be between"):
            ConfigurableController(
                patient=patient, meal_schedule=meal_schedule, correction_factor=2.0
            )

    def test_pre_bolus_too_high(self, patient, meal_schedule):
        """Verify error when pre-bolus time exceeds maximum."""
        from src.bgc_providers.simglucose_provider import ConfigurableController

        with pytest.raises(ValueError, match="pre_bolus_minutes must be between"):
            ConfigurableController(
                patient=patient, meal_schedule=meal_schedule, pre_bolus_minutes=60
            )

    def test_valid_custom_parameters(self, patient, meal_schedule):
        """Verify valid custom parameters are accepted."""
        from src.bgc_providers.simglucose_provider import ConfigurableController

        ctrl = ConfigurableController(
            patient=patient,
            meal_schedule=meal_schedule,
            basal_rate=1.5,
            target_glucose=120.0,
            carb_ratio=12.0,
            correction_factor=50.0,
            pre_bolus_minutes=15,
        )

        # Verify parameters were applied
        assert abs(ctrl._basal_rate - 1.5 / 60) < 1e-6  # Converted to U/min
        assert ctrl._target_glucose == 120.0
        assert ctrl._carb_ratio == 12.0
        assert ctrl._correction_factor == 50.0
        assert ctrl._pre_bolus_minutes == 15


class TestConfigurableControllerCustomBehavior:
    """Test custom parameter effects on insulin delivery."""

    @pytest.fixture
    def patient(self):
        """Create a test patient."""
        from simglucose.patient.t1dpatient import T1DPatient

        return T1DPatient.withName("adult#001")

    @pytest.fixture
    def meal_schedule(self):
        """Default meal schedule for testing."""
        return [(7, 45), (12, 70)]

    def test_custom_basal_rate_used(self, patient, meal_schedule):
        """Verify custom basal rate overrides patient default."""
        from src.bgc_providers.simglucose_provider import ConfigurableController

        custom_basal = 2.0  # U/hr
        ctrl = ConfigurableController(
            patient=patient, meal_schedule=meal_schedule, basal_rate=custom_basal
        )

        obs = MockObservation(CGM=120.0)
        info = {"patient_name": "adult#001", "meal": 0, "sample_time": 3}
        action = ctrl.policy(obs, 0, False, **info)

        expected_basal = custom_basal / 60  # Convert to U/min
        assert abs(action.basal - expected_basal) < 1e-6

    def test_custom_carb_ratio_affects_bolus(self, patient, meal_schedule):
        """Verify custom carb ratio changes meal bolus size."""
        from src.bgc_providers.simglucose_provider import ConfigurableController

        # CR = 5 means 5g carbs per unit (aggressive)
        ctrl_aggressive = ConfigurableController(
            patient=patient, meal_schedule=meal_schedule, carb_ratio=5.0
        )
        # CR = 20 means 20g carbs per unit (conservative)
        ctrl_conservative = ConfigurableController(
            patient=patient, meal_schedule=meal_schedule, carb_ratio=20.0
        )

        obs = MockObservation(CGM=120.0)  # Normal glucose
        info = {"patient_name": "adult#001", "meal": 20, "sample_time": 3}  # 60g meal

        action_aggressive = ctrl_aggressive.policy(obs, 0, False, **info)
        action_conservative = ctrl_conservative.policy(obs, 0, False, **info)

        # Aggressive should give more insulin
        assert action_aggressive.bolus > action_conservative.bolus

    def test_custom_target_affects_correction(self, patient, meal_schedule):
        """Verify custom target glucose changes correction bolus.

        Note: Correction is only applied at meal time (BBController behavior),
        so we need meal > 0 to see the correction difference.
        """
        from src.bgc_providers.simglucose_provider import ConfigurableController

        # Lower target = more correction
        ctrl_tight = ConfigurableController(
            patient=patient, meal_schedule=meal_schedule, target_glucose=100.0
        )
        # Higher target = less correction
        ctrl_loose = ConfigurableController(
            patient=patient, meal_schedule=meal_schedule, target_glucose=150.0
        )

        obs = MockObservation(CGM=200.0)  # High glucose
        # Need meal > 0 for correction to be applied (BBController behavior)
        info = {"patient_name": "adult#001", "meal": 10, "sample_time": 3}

        action_tight = ctrl_tight.policy(obs, 0, False, **info)
        action_loose = ctrl_loose.policy(obs, 0, False, **info)

        # Tighter target should give more correction insulin
        # Both have same meal bolus, but tight has larger correction
        assert action_tight.bolus > action_loose.bolus
