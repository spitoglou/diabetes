# Design: Configurable Insulin Controller

## Overview

This document describes the algorithmic and clinical foundations for each configurable insulin parameter, and the architectural approach for implementing the `ConfigurableController`.

## Architecture

### Controller Hierarchy

```
Controller (simglucose base)
├── NoInsulinController        # Existing: returns Action(basal=0, bolus=0)
├── BasalOnlyController        # Existing: returns Action(basal=rate, bolus=0)
├── BBController               # Simglucose built-in: fixed parameters
└── ConfigurableController     # NEW: user-overridable parameters
```

The `ConfigurableController` will replace both `BasalOnlyController` and `BBController` usage when any parameter is customized. It falls back to patient-specific defaults from simglucose's quest CSV when parameters are not specified.

### Parameter Resolution Order

```
CLI option > Environment variable > Patient-specific default > Fallback default
```

## Parameter Documentation

### 1. Basal Rate (`--basal-rate`)

#### Clinical Purpose
Basal insulin is the continuous background insulin that controls blood glucose between meals and during sleep. It counteracts the liver's constant glucose release (hepatic glucose output) and maintains glucose stability when not eating.

#### Algorithm
The default basal rate is derived from the patient's steady-state insulin requirement:

```
basal_rate = u2ss × BW / 6000
```

Where:
- `u2ss` = steady-state insulin concentration (pmol/L/kg) - patient-specific parameter
- `BW` = body weight (kg) - patient-specific parameter  
- `6000` = conversion factor to U/min

**Unit conversion:**
- Internal: U/min (simglucose native)
- User-facing: U/hr (clinical standard)
- Conversion: `U/hr = U/min × 60`

#### Clinical Context
- **Typical adult range:** 0.5-2.0 U/hr
- **Typical pediatric range:** 0.2-1.0 U/hr
- **Rule of thumb:** Total daily basal ≈ 40-50% of Total Daily Dose (TDD)
- **Adjustment:** Increase if fasting glucose consistently high; decrease if overnight lows occur

#### Validation
- Minimum: 0.0 U/hr (equivalent to `none` mode)
- Maximum: 5.0 U/hr (safety cap - higher values are clinically rare)
- Warning: Values outside 0.3-3.0 U/hr trigger a warning log

---

### 2. Target Glucose (`--target-glucose`)

#### Clinical Purpose
The target glucose is the blood glucose concentration that correction boluses aim to achieve. When current glucose exceeds this target, a correction dose is calculated to bring glucose down to this level.

#### Algorithm
Used in the correction bolus calculation:

```
correction_bolus = (current_glucose - target_glucose) / CF
```

Only applied when `current_glucose > correction_threshold` (typically target + 10 mg/dL or a fixed threshold like 150 mg/dL).

#### Clinical Context
- **Typical range:** 100-150 mg/dL
- **Tighter control:** 100-120 mg/dL (higher hypoglycemia risk)
- **Conservative:** 130-150 mg/dL (lower hypoglycemia risk, used for elderly or hypoglycemia-unaware patients)
- **Pregnancy:** 90-110 mg/dL (tighter targets)

#### Validation
- Minimum: 70 mg/dL (below this risks hypoglycemia)
- Maximum: 200 mg/dL (above this provides no meaningful correction)
- Default: 140 mg/dL (simglucose default)

---

### 3. Carbohydrate Ratio (`--carb-ratio`)

#### Clinical Purpose
The carbohydrate ratio (CR, also called I:C ratio or insulin-to-carb ratio) defines how many grams of carbohydrates are covered by one unit of insulin. It determines the meal bolus size.

#### Algorithm
```
meal_bolus = carbohydrates_consumed / CR
```

Where:
- `carbohydrates_consumed` = grams of carbs in the meal
- `CR` = carb ratio (g/U)

**Example:** 60g carb meal with CR of 10 g/U → 60/10 = 6 units bolus

#### Clinical Context
- **Typical adult range:** 5-25 g/U
- **Insulin sensitive (children):** 15-30 g/U
- **Insulin resistant:** 3-8 g/U
- **Time-of-day variation:** Many patients need lower CR (more insulin) at breakfast due to dawn phenomenon

#### Relationship to Other Parameters
- **500 Rule (approximation):** CR ≈ 500 / TDD
  - Example: TDD of 50U → CR ≈ 500/50 = 10 g/U
- Lower CR = more insulin per carb = more aggressive dosing
- Higher CR = less insulin per carb = more conservative dosing

#### Validation
- Minimum: 1 g/U (extremely insulin resistant - rare)
- Maximum: 50 g/U (extremely insulin sensitive - pediatric/honeymoon phase)
- Default: Patient-specific from quest CSV

---

### 4. Correction Factor (`--correction-factor`)

#### Clinical Purpose
The correction factor (CF, also called Insulin Sensitivity Factor or ISF) defines how much one unit of insulin will lower blood glucose. It determines correction bolus magnitude.

#### Algorithm
```
correction_bolus = (current_glucose - target_glucose) / CF
```

Where:
- `current_glucose` = current CGM reading (mg/dL)
- `target_glucose` = target BG (mg/dL)
- `CF` = correction factor (mg/dL per U)

**Example:** Current BG 250 mg/dL, target 140 mg/dL, CF 50 mg/dL/U
- Correction = (250 - 140) / 50 = 2.2 units

#### Clinical Context
- **Typical adult range:** 20-100 mg/dL/U
- **Insulin sensitive:** 80-150 mg/dL/U
- **Insulin resistant:** 10-30 mg/dL/U

#### Relationship to Other Parameters
- **1800 Rule (approximation):** CF ≈ 1800 / TDD
  - Example: TDD of 50U → CF ≈ 1800/50 = 36 mg/dL/U
- **1700 Rule (alternative):** CF ≈ 1700 / TDD (more conservative)
- Lower CF = more insulin per mg/dL correction = more aggressive
- Higher CF = less insulin per mg/dL correction = more conservative

#### Validation
- Minimum: 5 mg/dL/U (extremely insulin resistant)
- Maximum: 200 mg/dL/U (extremely insulin sensitive)
- Default: Patient-specific from quest CSV

---

### 5. Pre-Bolus Time (`--pre-bolus-minutes`)

#### Clinical Purpose
Pre-bolusing means delivering the meal bolus before eating, typically 15-30 minutes prior. This allows insulin to start working before glucose from food enters the bloodstream, reducing post-meal glucose spikes.

#### Algorithm
The controller tracks the meal schedule and delivers boluses early:

```python
for each simulation step:
    current_sim_time = get_current_simulation_time()
    
    for (meal_hour, meal_carbs) in meal_schedule:
        meal_time = current_date + meal_hour
        bolus_time = meal_time - pre_bolus_minutes
        
        if current_sim_time >= bolus_time and meal not yet bolused:
            deliver_bolus(meal_carbs / CR)
            mark_meal_as_bolused(meal_hour)
```

**Key behaviors:**
- Bolus is delivered `pre_bolus_minutes` before scheduled meal time
- No bolus is delivered when the actual meal occurs (already pre-bolused)
- Correction boluses are still delivered at meal time based on current glucose

#### Clinical Context
- **No pre-bolus (0 min):** Bolus at meal time - simpler but higher post-meal spikes
- **Moderate (10-15 min):** Good balance of efficacy and practicality
- **Aggressive (20-30 min):** Optimal spike reduction but requires meal timing precision
- **Risk:** If meal is delayed after pre-bolus, hypoglycemia can occur

#### Insulin Action Timing (Rapid-Acting Analogs)
- **Onset:** 10-15 minutes
- **Peak:** 60-90 minutes  
- **Duration:** 3-5 hours

Pre-bolusing aligns insulin onset with carbohydrate absorption onset (~15-30 min for most foods).

#### Validation
- Minimum: 0 minutes (no pre-bolus, bolus at meal time)
- Maximum: 45 minutes (longer pre-bolus rarely beneficial, increases hypo risk)
- Default: 0 minutes (matches simglucose BBController behavior)

---

## Implementation Notes

### ConfigurableController Class

```python
class ConfigurableController(Controller):
    def __init__(
        self,
        patient: T1DPatient,
        meal_schedule: list[tuple[int, int]],
        basal_rate: float | None = None,        # U/hr, None = use patient default
        target_glucose: float = 140.0,          # mg/dL
        carb_ratio: float | None = None,        # g/U, None = use patient default
        correction_factor: float | None = None, # mg/dL/U, None = use patient default
        pre_bolus_minutes: int = 0,             # minutes before meal
    ):
        ...
```

### State Tracking for Pre-Bolus

The controller must track:
1. Current simulation time (from `step.info['time']`)
2. Which meals have been pre-bolused (to avoid double-dosing)
3. Simulation sample time (3 minutes in simglucose)

### Settings Integration

New settings in `config/settings.py`:
```python
# Insulin Controller Configuration
SIMGLUCOSE_BASAL_RATE: float | None = None        # U/hr override
SIMGLUCOSE_TARGET_GLUCOSE: float = 140.0          # mg/dL
SIMGLUCOSE_CARB_RATIO: float | None = None        # g/U override
SIMGLUCOSE_CORRECTION_FACTOR: float | None = None # mg/dL/U override
SIMGLUCOSE_PRE_BOLUS_MINUTES: int = 0             # minutes
```

### CLI Integration

```bash
uv run python cli.py serve simglucose-client \
    --insulin-mode basal-bolus \
    --basal-rate 1.2 \
    --target-glucose 120 \
    --carb-ratio 12 \
    --correction-factor 50 \
    --pre-bolus-minutes 15
```

When any of these options are provided, the `ConfigurableController` is used instead of `BasalOnlyController` or `BBController`.

---

## Validation: Equivalence with BBController

### Requirement
When `ConfigurableController` is instantiated with default parameters (no overrides), it MUST produce identical insulin actions to simglucose's `BBController` for the same patient and scenario.

### Validation Test

```python
def test_configurable_controller_matches_bbcontroller():
    """
    Verify ConfigurableController with defaults produces identical 
    actions to BBController for the same simulation.
    """
    from simglucose.controller.basal_bolus_ctrller import BBController
    from src.bgc_providers.simglucose_provider import ConfigurableController
    
    patient = T1DPatient.withName('adult#001')
    meal_schedule = [(7, 45), (12, 70), (18, 80)]
    
    bb_ctrl = BBController(target=140)
    cfg_ctrl = ConfigurableController(
        patient=patient,
        meal_schedule=meal_schedule,
        # All defaults - no overrides
    )
    
    # Run both controllers through identical scenarios
    test_cases = [
        # (observation_cgm, meal_g_per_min, expected_match)
        (120.0, 0, True),    # No meal, normal glucose
        (180.0, 0, True),    # No meal, high glucose (correction)
        (120.0, 15, True),   # Meal present (15g/min for 3min = 45g)
        (200.0, 20, True),   # Meal + high glucose
    ]
    
    for cgm, meal, _ in test_cases:
        obs = MockObservation(CGM=cgm)
        info = {'patient_name': 'adult#001', 'meal': meal, 'sample_time': 3}
        
        bb_action = bb_ctrl.policy(obs, 0, False, **info)
        cfg_action = cfg_ctrl.policy(obs, 0, False, **info)
        
        assert abs(bb_action.basal - cfg_action.basal) < 1e-6, \
            f"Basal mismatch: BB={bb_action.basal}, Cfg={cfg_action.basal}"
        assert abs(bb_action.bolus - cfg_action.bolus) < 1e-6, \
            f"Bolus mismatch: BB={bb_action.bolus}, Cfg={cfg_action.bolus}"
```

### What Must Match

| Component | BBController Source | ConfigurableController Must Match |
|-----------|---------------------|-----------------------------------|
| Basal rate | `u2ss * BW / 6000` | Same formula when `basal_rate=None` |
| Target glucose | `self.target` (default 140) | Same default |
| Carb ratio | `quest.CR.values` lookup | Same CSV lookup when `carb_ratio=None` |
| Correction factor | `quest.CF.values` lookup | Same CSV lookup when `correction_factor=None` |
| Correction threshold | `glucose > 150` | Same threshold |
| Bolus formula | `(meal * sample_time) / CR + correction` | Identical formula |

### Pre-Bolus Equivalence

When `pre_bolus_minutes=0` (default), bolus timing MUST match BBController:
- Bolus delivered when `meal > 0` in observation
- No lookahead behavior

### Test Coverage

The validation task in `tasks.md` includes:
1. Unit test comparing actions for 100+ random scenarios
2. Integration test running full 24-hour simulation comparing glucose traces
3. Statistical validation: glucose RMSE between controllers < 0.1 mg/dL
