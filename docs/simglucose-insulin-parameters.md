# Configurable Insulin Parameters for Simglucose Simulation

This document provides comprehensive documentation of the configurable insulin parameters available in the `simglucose-client` command, including their clinical significance, algorithmic implementation, and validation results.

## Table of Contents

1. [Overview](#overview)
2. [Parameter Reference](#parameter-reference)
   - [Basal Rate](#basal-rate)
   - [Target Glucose](#target-glucose)
   - [Carbohydrate Ratio (CR)](#carbohydrate-ratio-cr)
   - [Correction Factor (CF)](#correction-factor-cf)
   - [Pre-Bolus Time](#pre-bolus-time)
3. [Simglucose Simulation Architecture](#simglucose-simulation-architecture)
4. [Clinical Validation Results](#clinical-validation-results)
5. [Usage Examples](#usage-examples)
6. [Parameter Relationships](#parameter-relationships)

---

## Overview

The diabetes prediction system includes a synthetic CGM data generator based on the [simglucose](https://github.com/jxx123/simglucose) library, which implements the FDA-approved UVA/Padova Type 1 Diabetes Metabolic Simulator. This provides realistic glucose dynamics for testing and development.

The `ConfigurableController` class allows customization of insulin delivery parameters, enabling simulation of various therapy scenarios:

- **Research**: Test prediction models against different insulin regimens
- **Education**: Demonstrate how insulin parameters affect glucose control
- **Development**: Generate diverse test data for model training
- **Clinical Scenarios**: Simulate pump settings mismatches or therapy adjustments

### Quick Reference

| Parameter | CLI Option | Default | Range | Unit |
|-----------|------------|---------|-------|------|
| Basal Rate | `--basal-rate` / `-b` | Patient-derived | 0.0 - 5.0 | U/hr |
| Target Glucose | `--target-glucose` / `-t` | 140 | 70 - 200 | mg/dL |
| Carb Ratio | `--carb-ratio` / `-c` | Patient-derived | 1 - 50 | g/U |
| Correction Factor | `--correction-factor` / `-f` | Patient-derived | 5 - 200 | mg/dL/U |
| Pre-Bolus Time | `--pre-bolus-minutes` / `-B` | 0 | 0 - 45 | minutes |

---

## Parameter Reference

### Basal Rate

#### Clinical Significance

Basal insulin is the continuous background insulin that controls blood glucose between meals and during sleep. It counteracts the liver's constant glucose release (hepatic glucose output) and maintains glucose stability when not eating.

In real Type 1 Diabetes management:
- Basal insulin typically accounts for 40-50% of Total Daily Dose (TDD)
- Rates vary by time of day (often higher in early morning due to "dawn phenomenon")
- Incorrect basal rates cause fasting hyperglycemia (too low) or hypoglycemia (too high)

#### Algorithm

The default basal rate is derived from patient-specific steady-state parameters:

```
basal_rate (U/min) = u2ss × BW / 6000
```

Where:
- `u2ss` = steady-state insulin concentration (pmol/L/kg) - patient-specific
- `BW` = body weight (kg) - patient-specific
- `6000` = conversion factor

**Unit Conversion:**
- Internal calculation: U/min (simglucose native unit)
- User-facing: U/hr (clinical standard)
- Conversion: `U/hr = U/min × 60`

#### Typical Values

| Population | Typical Range | Notes |
|------------|---------------|-------|
| Adults | 0.5 - 2.0 U/hr | Higher with insulin resistance |
| Adolescents | 0.4 - 1.5 U/hr | Variable due to growth hormones |
| Children | 0.2 - 1.0 U/hr | Lower body weight |

#### CLI Usage

```bash
# Use patient default (calculated from model parameters)
uv run python cli.py serve simglucose-client -i basal-bolus

# Override with custom rate
uv run python cli.py serve simglucose-client -i basal-bolus --basal-rate 1.5

# Test insulin-sensitive scenario
uv run python cli.py serve simglucose-client -i basal-bolus -b 0.8

# Test insulin-resistant scenario
uv run python cli.py serve simglucose-client -i basal-bolus -b 2.5
```

---

### Target Glucose

#### Clinical Significance

The target glucose is the blood glucose concentration that correction boluses aim to achieve. When current glucose exceeds a threshold (150 mg/dL in this implementation), a correction dose is calculated to bring glucose down to the target.

In real diabetes management:
- Tighter targets (100-120 mg/dL) improve average glucose but increase hypoglycemia risk
- Conservative targets (130-150 mg/dL) reduce hypoglycemia risk but allow higher averages
- Targets are often personalized based on hypoglycemia awareness, age, and comorbidities

#### Algorithm

Correction bolus is calculated when glucose exceeds the correction threshold (150 mg/dL):

```
correction_bolus (U) = (current_glucose - target_glucose) / CF
```

**Important:** Following BBController behavior, correction boluses are only applied at meal times, not as standalone corrections between meals.

#### Typical Values

| Scenario | Target Range | Clinical Context |
|----------|--------------|------------------|
| Tight Control | 100 - 120 mg/dL | Young, healthy, hypoglycemia-aware |
| Standard | 120 - 140 mg/dL | Most adults |
| Conservative | 140 - 160 mg/dL | Elderly, hypoglycemia-unaware |
| Pregnancy | 90 - 110 mg/dL | Stricter targets for fetal health |

#### CLI Usage

```bash
# Default target (140 mg/dL)
uv run python cli.py serve simglucose-client -i basal-bolus

# Tighter control
uv run python cli.py serve simglucose-client -i basal-bolus --target-glucose 110

# More conservative
uv run python cli.py serve simglucose-client -i basal-bolus -t 150
```

---

### Carbohydrate Ratio (CR)

#### Clinical Significance

The carbohydrate ratio (CR), also called insulin-to-carb ratio (I:C), defines how many grams of carbohydrates are covered by one unit of insulin. It determines meal bolus size and is one of the most critical parameters for post-meal glucose control.

In real diabetes management:
- CR varies significantly between individuals (5-30 g/U range)
- Often varies by time of day (lower at breakfast due to dawn phenomenon)
- Incorrect CR causes post-meal hyperglycemia (too high) or hypoglycemia (too low)

#### Algorithm

Meal bolus is calculated when carbohydrates are consumed:

```
meal_bolus (U) = carbs_consumed (g) / CR (g/U)
```

**Example:** 60g carb meal with CR of 10 g/U → 60 / 10 = 6 units

#### Approximation Rule

The "500 Rule" provides a starting estimate:

```
CR ≈ 500 / TDD
```

Where TDD = Total Daily Dose of insulin

**Example:** TDD of 50 units → CR ≈ 500 / 50 = 10 g/U

#### Typical Values

| Insulin Sensitivity | CR Range | Interpretation |
|--------------------|----------|----------------|
| Very Sensitive | 20 - 30 g/U | Less insulin needed per carb |
| Normal | 10 - 15 g/U | Typical adult range |
| Resistant | 5 - 8 g/U | More insulin needed per carb |
| Highly Resistant | 3 - 5 g/U | Significant insulin resistance |

#### CLI Usage

```bash
# Use patient default from simglucose quest CSV
uv run python cli.py serve simglucose-client -i basal-bolus

# More aggressive (lower CR = more insulin per carb)
uv run python cli.py serve simglucose-client -i basal-bolus --carb-ratio 8

# More conservative (higher CR = less insulin per carb)
uv run python cli.py serve simglucose-client -i basal-bolus -c 18
```

---

### Correction Factor (CF)

#### Clinical Significance

The correction factor (CF), also called Insulin Sensitivity Factor (ISF), defines how much one unit of insulin will lower blood glucose. It determines correction bolus magnitude when glucose is above target.

In real diabetes management:
- CF indicates overall insulin sensitivity
- Higher CF = more sensitive (glucose drops more per unit)
- Lower CF = more resistant (glucose drops less per unit)
- Can vary by time of day and activity level

#### Algorithm

Correction bolus is calculated when glucose exceeds threshold (150 mg/dL) at meal time:

```
correction_bolus (U) = (current_glucose - target_glucose) / CF
```

**Example:** Current BG 250 mg/dL, target 140 mg/dL, CF 50 mg/dL/U
- Correction = (250 - 140) / 50 = 2.2 units

#### Approximation Rule

The "1800 Rule" provides a starting estimate:

```
CF ≈ 1800 / TDD
```

**Example:** TDD of 50 units → CF ≈ 1800 / 50 = 36 mg/dL/U

#### Typical Values

| Insulin Sensitivity | CF Range | Interpretation |
|--------------------|----------|----------------|
| Very Sensitive | 80 - 150 mg/dL/U | Large BG drop per unit |
| Normal | 30 - 60 mg/dL/U | Typical adult range |
| Resistant | 15 - 30 mg/dL/U | Smaller BG drop per unit |
| Highly Resistant | 5 - 15 mg/dL/U | Significant resistance |

#### CLI Usage

```bash
# Use patient default
uv run python cli.py serve simglucose-client -i basal-bolus

# More sensitive (higher CF)
uv run python cli.py serve simglucose-client -i basal-bolus --correction-factor 60

# More resistant (lower CF)
uv run python cli.py serve simglucose-client -i basal-bolus -f 25
```

---

### Pre-Bolus Time

#### Clinical Significance

Pre-bolusing means delivering the meal bolus before eating, typically 15-30 minutes prior. This allows insulin to start working before glucose from food enters the bloodstream, reducing post-meal glucose spikes.

In real diabetes management:
- Rapid-acting insulin analogs have onset of 10-15 minutes, peak at 60-90 minutes
- Carbohydrate absorption begins 15-30 minutes after eating
- Pre-bolusing aligns insulin action with glucose absorption
- Risk: If meal is delayed after pre-bolus, hypoglycemia can occur

#### Algorithm

The controller tracks the meal schedule and delivers boluses early:

```python
for each simulation step:
    current_time = get_simulation_time()
    
    for (meal_hour, meal_carbs) in meal_schedule:
        bolus_time = meal_hour - pre_bolus_minutes
        
        if current_time >= bolus_time and meal not yet bolused:
            deliver_bolus(meal_carbs / CR)
            mark_meal_as_bolused()
```

**Key Behaviors:**
- Bolus delivered `pre_bolus_minutes` before scheduled meal time
- No duplicate bolus at actual meal time (already pre-bolused)
- Correction boluses still applied at meal time based on current glucose

#### Typical Values

| Pre-Bolus Time | Use Case |
|----------------|----------|
| 0 min | Default, bolus at meal time |
| 10-15 min | Moderate pre-bolus, good for most situations |
| 20-30 min | Aggressive pre-bolus, optimal spike reduction |
| 30-45 min | Extended pre-bolus, high-glycemic meals |

#### CLI Usage

```bash
# No pre-bolus (default, matches BBController)
uv run python cli.py serve simglucose-client -i basal-bolus

# 15-minute pre-bolus
uv run python cli.py serve simglucose-client -i basal-bolus --pre-bolus-minutes 15

# 30-minute pre-bolus for high-carb meals
uv run python cli.py serve simglucose-client -i basal-bolus -B 30
```

---

## Simglucose Simulation Architecture

### UVA/Padova Model Overview

The simglucose library implements the UVA/Padova Type 1 Diabetes Metabolic Simulator, an FDA-approved computer model that simulates:

1. **Glucose Subsystem**: Plasma and tissue glucose dynamics
2. **Insulin Subsystem**: Plasma insulin kinetics and action
3. **Meal Absorption**: Gut glucose absorption from carbohydrates
4. **Insulin Delivery**: Subcutaneous insulin pharmacokinetics

### Virtual Patients

The simulator includes 30 virtual patients with realistic physiological variation:

| Group | Patients | Characteristics |
|-------|----------|-----------------|
| Adults | adult#001 - adult#010 | Age 26-68, varied insulin sensitivity |
| Adolescents | adolescent#001 - adolescent#010 | Age 14-19, growth hormone effects |
| Children | child#001 - child#010 | Age 7-12, high insulin sensitivity |

Each patient has unique parameters for:
- Body weight (BW)
- Steady-state insulin (u2ss)
- Carb ratio (CR)
- Correction factor (CF)
- Total daily insulin (TDI)

### Controller Hierarchy

```
Controller (simglucose base)
├── NoInsulinController        # Open-loop, no insulin
├── BasalOnlyController        # Constant basal rate only
├── BBController               # Simglucose built-in basal-bolus
└── ConfigurableController     # Custom parameters (this implementation)
```

The `ConfigurableController` is used when:
- `--insulin-mode basal-bolus` is specified, OR
- Any insulin parameter override is provided

### Simulation Time Steps

- **Step interval**: 3 minutes (simglucose default)
- **CGM readings**: Every 3 simulation minutes
- **Real-time streaming**: Configurable delay between readings (default 20 seconds)
- **Fast-forward**: Can advance simulation to current time of day on startup

---

## Clinical Validation Results

### Test Methodology

To validate clinical realism, we ran 24-hour simulations (480 steps) with different parameter configurations using:

- **Patient**: adult#001
- **Seed**: 42 (reproducible)
- **Meal Schedule**: 7am (45g), 12pm (70g), 4pm (15g), 6pm (80g), 11pm (10g)

### Results Summary

| Scenario | Avg (mg/dL) | Min | Max | TIR% | Hypo% | Hyper% |
|----------|-------------|-----|-----|------|-------|--------|
| **Default (patient params)** | 144.9 | 86.7 | 204.9 | 94.4% | 0.0% | 5.6% |
| Higher basal (2.0 U/hr) | 112.0 | 50.9 | 159.0 | 96.7% | 3.3% | 0.0% |
| Lower basal (0.5 U/hr) | 165.2 | 117.0 | 236.2 | 69.6% | 0.0% | 30.4% |
| Aggressive CR=5 | 113.0 | 39.0 | 175.4 | 87.3% | 12.7% | 0.0% |
| Conservative CR=20 | 154.8 | 104.4 | 226.3 | 87.9% | 0.0% | 12.1% |
| Tight target=100 | 138.8 | 63.3 | 204.9 | 92.3% | 2.1% | 5.6% |
| Loose target=160 | 148.5 | 100.7 | 204.9 | 92.7% | 0.0% | 7.3% |

**Metrics:**
- **TIR (Time in Range)**: Percentage of readings between 70-180 mg/dL
- **Hypo%**: Percentage of readings below 70 mg/dL (hypoglycemia)
- **Hyper%**: Percentage of readings above 180 mg/dL (hyperglycemia)

### Clinical Observations

#### 1. Basal Rate Effects

**Higher Basal (2.0 U/hr):**
- Average glucose dropped from 144.9 to 112.0 mg/dL
- Eliminated hyperglycemia (0% > 180 mg/dL)
- Introduced hypoglycemia risk (3.3% < 70 mg/dL, min=50.9)
- **Clinical interpretation**: Overly aggressive basal causes lows

**Lower Basal (0.5 U/hr):**
- Average glucose rose from 144.9 to 165.2 mg/dL
- TIR dropped from 94.4% to 69.6%
- Significant hyperglycemia (30.4% > 180 mg/dL)
- **Clinical interpretation**: Insufficient basal leads to poor control

#### 2. Carb Ratio Effects

**Aggressive CR=5 (more insulin per carb):**
- Lower average (113.0 mg/dL)
- **Dangerous hypoglycemia**: 12.7% < 70 mg/dL, minimum 39.0 mg/dL
- Despite lower average, TIR is worse due to lows
- **Clinical interpretation**: Overly aggressive CR is dangerous

**Conservative CR=20 (less insulin per carb):**
- Higher average (154.8 mg/dL)
- Post-meal hyperglycemia (12.1% > 180 mg/dL)
- No hypoglycemia
- **Clinical interpretation**: Under-dosing allows post-meal spikes

#### 3. Target Glucose Effects

**Tight Target (100 mg/dL):**
- Lower average (138.8 vs 144.9 mg/dL)
- Introduced some hypoglycemia (2.1%)
- **Clinical interpretation**: Tighter targets increase hypo risk

**Loose Target (160 mg/dL):**
- Higher average (148.5 mg/dL)
- Zero hypoglycemia
- Slightly more hyperglycemia (7.3% vs 5.6%)
- **Clinical interpretation**: Conservative targets are safer but allow higher glucose

### Validation Conclusions

All parameter effects match expected clinical diabetes physiology:

| Check | Result | Clinical Basis |
|-------|--------|----------------|
| Higher basal → lower glucose | PASS | More insulin reduces glucose |
| Lower basal → higher glucose | PASS | Less insulin allows glucose rise |
| Lower CR → lower glucose | PASS | More meal insulin covers carbs better |
| Higher CR → higher glucose | PASS | Less meal insulin allows post-meal spikes |
| Tighter target → lower avg | PASS | More aggressive corrections |
| Aggressive dosing → hypoglycemia | PASS | Over-insulinization causes lows |

The implementation correctly models the **TIR vs. hypoglycemia trade-off** that is fundamental to real diabetes management.

---

## Detailed 24-Hour Timetables

The following timetables show glucose values every 30 minutes for a full 24-hour simulation, with meal times and insulin effects clearly visible.

**Simulation Parameters:**
- Patient: adult#001
- Seed: 42 (reproducible)
- Meal Schedule: 7:00 (45g), 12:00 (70g), 16:00 (15g), 18:00 (80g), 23:00 (10g)
- Basal-bolus insulin mode

**Zone Legend:**
- **In Range**: 70-180 mg/dL (target)
- **HYPO**: < 70 mg/dL (hypoglycemia - dangerous)
- **HIGH**: > 180 mg/dL (hyperglycemia)

### Scenario 1: Default (Patient Parameters)

**Parameters:** CR=10 g/U, CF=8.8 mg/dL/U, Basal=1.27 U/hr, Target=140 mg/dL

| Time | Glucose (mg/dL) | Meal (g) | Zone | Notes |
|------|-----------------|----------|------|-------|
| 00:00 |  142.3 |        - | In Range |  |
| 00:30 |  152.0 |        - | In Range |  |
| 01:00 |  143.6 |        - | In Range |  |
| 01:30 |  159.6 |        - | In Range |  |
| 02:00 |  146.2 |        - | In Range |  |
| 02:30 |  136.8 |        - | In Range |  |
| 03:00 |  134.3 |        - | In Range |  |
| 03:30 |  117.5 |        - | In Range |  |
| 04:00 |  121.5 |        - | In Range |  |
| 04:30 |  119.8 |        - | In Range |  |
| 05:00 |  132.8 |        - | In Range |  |
| 05:30 |  132.1 |        - | In Range |  |
| 06:00 |  129.6 |        - | In Range |  |
| 06:30 |  127.8 |        - | In Range |  |
| **07:00** |  **129.5** |       **45** | In Range | **Breakfast: 45g carbs** |
| 07:30 |  150.6 |        - | In Range | Post-meal rise |
| 08:00 |  174.3 |        - | In Range | Peak post-breakfast |
| 08:30 |  177.9 |        - | In Range |  |
| 09:00 |  166.0 |        - | In Range | Returning to baseline |
| 09:30 |  147.7 |        - | In Range |  |
| 10:00 |  157.3 |        - | In Range |  |
| 10:30 |  155.2 |        - | In Range |  |
| 11:00 |  148.0 |        - | In Range |  |
| 11:30 |  152.1 |        - | In Range |  |
| **12:00** |  **149.0** |       **70** | In Range | **Lunch: 70g carbs** |
| 12:30 |  155.5 |        - | In Range | Post-meal rise |
| 13:00 |  186.6 |        - | HIGH | Peak post-lunch |
| 13:30 |  204.6 |        - | HIGH | Maximum excursion |
| 14:00 |  180.3 |        - | HIGH | Returning |
| 14:30 |  171.2 |        - | In Range |  |
| 15:00 |  155.4 |        - | In Range |  |
| 15:30 |  136.1 |        - | In Range |  |
| **16:00** |  **156.8** |       **15** | In Range | **Snack: 15g carbs** |
| 16:30 |  163.4 |        - | In Range | Minimal rise (small snack) |
| 17:00 |  156.6 |        - | In Range |  |
| 17:30 |  153.1 |        - | In Range |  |
| **18:00** |  **140.9** |       **80** | In Range | **Dinner: 80g carbs** |
| 18:30 |  126.0 |        - | In Range | Initial dip (bolus effect) |
| 19:00 |  167.6 |        - | In Range | Post-dinner rise |
| 19:30 |  160.6 |        - | In Range |  |
| 20:00 |  138.2 |        - | In Range |  |
| 20:30 |  137.6 |        - | In Range |  |
| 21:00 |  111.7 |        - | In Range |  |
| 21:30 |  118.0 |        - | In Range |  |
| 22:00 |  124.5 |        - | In Range |  |
| 22:30 |  133.0 |        - | In Range |  |
| **23:00** |  **110.2** |       **10** | In Range | **Bedtime snack: 10g carbs** |
| 23:30 |   92.6 |        - | In Range |  |

**Summary:** Avg=145.5 mg/dL | Min=92.6 | Max=204.6 | TIR=93.8% | Hypo=0.0%

**Clinical Assessment:** Good control overall. Post-lunch hyperglycemia (204.6 mg/dL) suggests CR may be slightly too conservative for large meals. No hypoglycemia.

---

### Scenario 2: Higher Basal Rate (2.0 U/hr)

**Parameters:** Basal=2.0 U/hr (vs default 1.27 U/hr), all other parameters default

| Time | Glucose (mg/dL) | Meal (g) | Zone | Notes |
|------|-----------------|----------|------|-------|
| 00:00 |  142.3 |        - | In Range |  |
| 00:30 |  152.0 |        - | In Range |  |
| 01:00 |  143.5 |        - | In Range |  |
| 01:30 |  159.0 |        - | In Range |  |
| 02:00 |  144.6 |        - | In Range |  |
| 02:30 |  133.7 |        - | In Range |  |
| 03:00 |  129.2 |        - | In Range |  |
| 03:30 |  109.9 |        - | In Range | Lower overnight |
| 04:00 |  111.2 |        - | In Range |  |
| 04:30 |  106.5 |        - | In Range |  |
| 05:00 |  116.5 |        - | In Range |  |
| 05:30 |  112.7 |        - | In Range |  |
| 06:00 |  107.2 |        - | In Range |  |
| 06:30 |  102.4 |        - | In Range |  |
| **07:00** |  **101.3** |       **45** | In Range | **Breakfast** (lower starting BG) |
| 07:30 |  119.5 |        - | In Range | Reduced post-meal rise |
| 08:00 |  140.6 |        - | In Range |  |
| 08:30 |  141.4 |        - | In Range | Peak much lower than default |
| 09:00 |  127.1 |        - | In Range |  |
| 09:30 |  107.0 |        - | In Range |  |
| 10:00 |  115.1 |        - | In Range |  |
| 10:30 |  111.8 |        - | In Range |  |
| 11:00 |  103.3 |        - | In Range |  |
| 11:30 |  105.7 |        - | In Range |  |
| **12:00** |  **100.8** |       **70** | In Range | **Lunch** |
| 12:30 |  106.0 |        - | In Range |  |
| 13:00 |  136.1 |        - | In Range | Much lower peak! |
| 13:30 |  152.9 |        - | In Range | No hyperglycemia |
| 14:00 |  128.1 |        - | In Range |  |
| 14:30 |  119.3 |        - | In Range |  |
| 15:00 |  104.3 |        - | In Range |  |
| 15:30 |   85.9 |        - | In Range | Getting lower |
| **16:00** |  **107.0** |       **15** | In Range | **Snack** |
| 16:30 |  113.2 |        - | In Range |  |
| 17:00 |  106.6 |        - | In Range |  |
| 17:30 |  104.6 |        - | In Range |  |
| **18:00** |  **94.8** |       **80** | In Range | **Dinner** |
| 18:30 |   82.7 |        - | In Range | Lower baseline |
| 19:00 |  126.6 |        - | In Range |  |
| 19:30 |  121.1 |        - | In Range |  |
| 20:00 |  100.2 |        - | In Range |  |
| 20:30 |  101.2 |        - | In Range |  |
| 21:00 |   76.9 |        - | In Range | Approaching low |
| 21:30 |   84.4 |        - | In Range |  |
| 22:00 |   91.4 |        - | In Range |  |
| 22:30 |   99.2 |        - | In Range |  |
| **23:00** |  **75.4** |       **10** | In Range | **Bedtime snack** |
| 23:30 |   **57.0** |        - | **HYPO** | **Hypoglycemia!** |

**Summary:** Avg=112.9 mg/dL | Min=57.0 | Max=159.0 | TIR=97.9% | Hypo=2.1%

**Clinical Assessment:** Excellent average glucose and eliminated hyperglycemia, but caused overnight hypoglycemia (57 mg/dL). This demonstrates the classic trade-off: more aggressive insulin improves average but increases hypo risk.

---

### Scenario 3: Lower Basal Rate (0.5 U/hr)

**Parameters:** Basal=0.5 U/hr (vs default 1.27 U/hr), all other parameters default

| Time | Glucose (mg/dL) | Meal (g) | Zone | Notes |
|------|-----------------|----------|------|-------|
| 00:00 |  142.3 |        - | In Range |  |
| 00:30 |  152.0 |        - | In Range |  |
| 01:00 |  143.8 |        - | In Range |  |
| 01:30 |  160.2 |        - | In Range |  |
| 02:00 |  147.9 |        - | In Range |  |
| 02:30 |  140.1 |        - | In Range |  |
| 03:00 |  139.9 |        - | In Range |  |
| 03:30 |  125.8 |        - | In Range |  |
| 04:00 |  133.1 |        - | In Range |  |
| 04:30 |  135.0 |        - | In Range |  |
| 05:00 |  151.9 |        - | In Range | Rising overnight |
| 05:30 |  155.2 |        - | In Range |  |
| 06:00 |  157.0 |        - | In Range |  |
| 06:30 |  159.6 |        - | In Range |  |
| **07:00** |  **165.7** |       **45** | In Range | **Breakfast** (high starting BG) |
| 07:30 |  190.9 |        - | **HIGH** | Rapid rise |
| 08:00 |  216.7 |        - | **HIGH** | Severe hyperglycemia |
| 08:30 |  219.8 |        - | **HIGH** |  |
| 09:00 |  205.8 |        - | **HIGH** |  |
| 09:30 |  185.1 |        - | **HIGH** |  |
| 10:00 |  192.7 |        - | **HIGH** |  |
| 10:30 |  189.5 |        - | **HIGH** |  |
| 11:00 |  182.1 |        - | **HIGH** |  |
| 11:30 |  187.0 |        - | **HIGH** |  |
| **12:00** |  **185.4** |       **70** | **HIGH** | **Lunch** (starting high) |
| 12:30 |  193.7 |        - | **HIGH** |  |
| 13:00 |  223.5 |        - | **HIGH** |  |
| 13:30 |  **236.2** |        - | **HIGH** | **Maximum: 236.2 mg/dL** |
| 14:00 |  204.8 |        - | **HIGH** |  |
| 14:30 |  188.9 |        - | **HIGH** |  |
| 15:00 |  167.8 |        - | In Range | Finally returning |
| 15:30 |  145.0 |        - | In Range |  |
| **16:00** |  **163.8** |       **15** | In Range | **Snack** |
| 16:30 |  169.8 |        - | In Range |  |
| 17:00 |  163.1 |        - | In Range |  |
| 17:30 |  159.9 |        - | In Range |  |
| **18:00** |  **148.2** |       **80** | In Range | **Dinner** |
| 18:30 |  134.2 |        - | In Range |  |
| 19:00 |  176.9 |        - | In Range |  |
| 19:30 |  171.8 |        - | In Range |  |
| 20:00 |  151.5 |        - | In Range |  |
| 20:30 |  152.8 |        - | In Range |  |
| 21:00 |  128.6 |        - | In Range |  |
| 21:30 |  136.6 |        - | In Range |  |
| 22:00 |  145.1 |        - | In Range |  |
| 22:30 |  156.4 |        - | In Range |  |
| **23:00** |  **136.7** |       **10** | In Range | **Bedtime snack** |
| 23:30 |  121.9 |        - | In Range |  |

**Summary:** Avg=165.4 mg/dL | Min=121.9 | Max=236.2 | TIR=68.8% | Hypo=0.0%

**Clinical Assessment:** Poor glycemic control. Insufficient basal leads to elevated fasting glucose and severe post-meal hyperglycemia. The patient would need significantly more insulin.

---

### Scenario 4: Aggressive Carb Ratio (CR=5 g/U)

**Parameters:** CR=5 g/U (vs default 10 g/U), all other parameters default

| Time | Glucose (mg/dL) | Meal (g) | Zone | Notes |
|------|-----------------|----------|------|-------|
| 00:00 |  142.3 |        - | In Range |  |
| 00:30 |  152.0 |        - | In Range |  |
| ... | ... | ... | ... | (same as default until first meal) |
| 06:30 |  127.8 |        - | In Range |  |
| **07:00** |  **129.5** |       **45** | In Range | **Breakfast: 45g → 9U bolus** (vs 4.5U default) |
| 07:30 |  150.4 |        - | In Range |  |
| 08:00 |  171.6 |        - | In Range | Lower peak than default |
| 08:30 |  169.7 |        - | In Range |  |
| 09:00 |  150.9 |        - | In Range |  |
| 09:30 |  126.3 |        - | In Range | Falls faster |
| 10:00 |  130.9 |        - | In Range |  |
| 10:30 |  125.1 |        - | In Range |  |
| 11:00 |  115.3 |        - | In Range |  |
| 11:30 |  117.4 |        - | In Range |  |
| **12:00** |  **113.0** |       **70** | In Range | **Lunch: 70g → 14U bolus** (vs 7U default) |
| 12:30 |  118.8 |        - | In Range |  |
| 13:00 |  146.9 |        - | In Range | Much lower post-lunch! |
| 13:30 |  158.4 |        - | In Range | No hyperglycemia |
| 14:00 |  127.1 |        - | In Range |  |
| 14:30 |  113.0 |        - | In Range |  |
| 15:00 |   94.7 |        - | In Range | Dropping |
| 15:30 |   74.6 |        - | In Range | Getting low |
| **16:00** |  **94.9** |       **15** | In Range | **Snack** |
| 16:30 |  100.8 |        - | In Range |  |
| 17:00 |   94.1 |        - | In Range |  |
| 17:30 |   92.1 |        - | In Range |  |
| **18:00** |  **82.7** |       **80** | In Range | **Dinner: 80g → 16U bolus** |
| 18:30 |   71.1 |        - | In Range | Near-hypo |
| 19:00 |  113.3 |        - | In Range |  |
| 19:30 |  103.2 |        - | In Range |  |
| 20:00 |   77.6 |        - | In Range |  |
| 20:30 |   76.1 |        - | In Range |  |
| 21:00 |   **51.1** |        - | **HYPO** | **Hypoglycemia** |
| 21:30 |   **58.9** |        - | **HYPO** |  |
| 22:00 |   **66.1** |        - | **HYPO** |  |
| 22:30 |   73.8 |        - | In Range |  |
| **23:00** |  **50.1** |       **10** | **HYPO** | **Bedtime snack** |
| 23:30 |   **39.0** |        - | **HYPO** | **Severe hypo: 39 mg/dL** |

**Summary:** Avg=114.0 mg/dL | Min=39.0 | Max=171.6 | TIR=89.6% | Hypo=10.4%

**Clinical Assessment:** DANGEROUS. While average glucose is lower and post-meal hyperglycemia is eliminated, the aggressive CR causes severe late-day hypoglycemia (39 mg/dL is medical emergency territory). This demonstrates why CR must be carefully individualized.

---

### Scenario 5: Conservative Carb Ratio (CR=20 g/U)

**Parameters:** CR=20 g/U (vs default 10 g/U), all other parameters default

| Time | Glucose (mg/dL) | Meal (g) | Zone | Notes |
|------|-----------------|----------|------|-------|
| 00:00 |  142.3 |        - | In Range |  |
| ... | ... | ... | ... | (same as default until first meal) |
| 06:30 |  127.8 |        - | In Range |  |
| **07:00** |  **129.5** |       **45** | In Range | **Breakfast: 45g → 2.25U bolus** (vs 4.5U default) |
| 07:30 |  150.7 |        - | In Range |  |
| 08:00 |  175.7 |        - | In Range |  |
| 08:30 |  **182.1** |        - | **HIGH** | Under-dosed, going high |
| 09:00 |  173.9 |        - | In Range |  |
| 09:30 |  159.4 |        - | In Range |  |
| 10:00 |  172.2 |        - | In Range |  |
| 10:30 |  172.6 |        - | In Range |  |
| 11:00 |  167.4 |        - | In Range |  |
| 11:30 |  172.8 |        - | In Range | Elevated baseline |
| **12:00** |  **170.7** |       **70** | In Range | **Lunch: 70g → 3.5U bolus** (vs 7U default) |
| 12:30 |  177.7 |        - | In Range |  |
| 13:00 |  **208.6** |        - | **HIGH** | Significant hyperglycemia |
| 13:30 |  **226.0** |        - | **HIGH** | Peak: 226 mg/dL |
| 14:00 |  **200.8** |        - | **HIGH** |  |
| 14:30 |  **190.5** |        - | **HIGH** |  |
| 15:00 |  173.6 |        - | In Range |  |
| 15:30 |  153.1 |        - | In Range |  |
| **16:00** |  **172.8** |       **15** | In Range | **Snack** |
| 16:30 |  178.5 |        - | In Range |  |
| 17:00 |  170.3 |        - | In Range |  |
| 17:30 |  164.7 |        - | In Range |  |
| **18:00** |  **150.0** |       **80** | In Range | **Dinner: 80g → 4U bolus** (vs 8U default) |
| 18:30 |  133.0 |        - | In Range |  |
| 19:00 |  173.8 |        - | In Range |  |
| 19:30 |  168.2 |        - | In Range |  |
| 20:00 |  148.1 |        - | In Range |  |
| 20:30 |  149.9 |        - | In Range |  |
| 21:00 |  125.7 |        - | In Range |  |
| 21:30 |  133.1 |        - | In Range |  |
| 22:00 |  140.5 |        - | In Range |  |
| 22:30 |  149.9 |        - | In Range |  |
| **23:00** |  **127.7** |       **10** | In Range | **Bedtime snack** |
| 23:30 |  110.3 |        - | In Range |  |

**Summary:** Avg=155.2 mg/dL | Min=110.3 | Max=226.0 | TIR=89.6% | Hypo=0.0%

**Clinical Assessment:** Safe but suboptimal. No hypoglycemia (which is good), but significant post-meal hyperglycemia (226 mg/dL after lunch). The patient would likely need a lower CR for better post-meal control, especially at lunch.

---

### Timetable Comparison Summary

| Scenario | Pre-Breakfast | Post-Breakfast Peak | Post-Lunch Peak | Late Evening | Hypo Events |
|----------|---------------|---------------------|-----------------|--------------|-------------|
| Default | 129.5 | 177.9 | 204.6 | 110.2 | None |
| High Basal | 101.3 | 141.4 | 152.9 | 75.4 | 1 (57.0) |
| Low Basal | 165.7 | 219.8 | 236.2 | 136.7 | None |
| Aggressive CR | 129.5 | 171.6 | 158.4 | 50.1 | 5 (min 39.0) |
| Conservative CR | 129.5 | 182.1 | 226.0 | 127.7 | None |

**Key Observations:**
1. **Basal affects fasting/overnight glucose** more than post-meal peaks
2. **CR affects post-meal glucose** directly and dramatically
3. **Over-aggressive insulin causes hypoglycemia** later (insulin stacking effect)
4. **Under-dosing is safer** but leads to hyperglycemia and poor long-term outcomes

---

## Usage Examples

### Basic Usage

```bash
# Start with default patient parameters
uv run python cli.py serve simglucose-client -i basal-bolus

# Use a different virtual patient
uv run python cli.py serve simglucose-client -i basal-bolus -p adolescent#005

# Reproducible simulation with seed
uv run python cli.py serve simglucose-client -i basal-bolus --seed 42
```

### Custom Insulin Parameters

```bash
# Simulate insulin-sensitive patient
uv run python cli.py serve simglucose-client -i basal-bolus \
    --basal-rate 0.8 \
    --carb-ratio 18 \
    --correction-factor 60

# Simulate insulin-resistant patient
uv run python cli.py serve simglucose-client -i basal-bolus \
    --basal-rate 2.5 \
    --carb-ratio 6 \
    --correction-factor 25

# Tight control with pre-bolusing
uv run python cli.py serve simglucose-client -i basal-bolus \
    --target-glucose 110 \
    --pre-bolus-minutes 20
```

### Research Scenarios

```bash
# Simulate pump failure (no insulin)
uv run python cli.py serve simglucose-client -i none

# Simulate missed boluses (basal only)
uv run python cli.py serve simglucose-client -i basal

# Simulate misconfigured pump (wrong CR)
uv run python cli.py serve simglucose-client -i basal-bolus --carb-ratio 25
```

### Environment Variables

All parameters can also be set via environment variables in `.env`:

```bash
# Simglucose Configuration
SIMGLUCOSE_PATIENT=adult#001
SIMGLUCOSE_SEED=42
SIMGLUCOSE_MEALS=7:45,12:70,16:15,18:80,23:10
SIMGLUCOSE_INSULIN_MODE=basal-bolus

# Configurable Insulin Parameters
SIMGLUCOSE_BASAL_RATE=1.2
SIMGLUCOSE_TARGET_GLUCOSE=130
SIMGLUCOSE_CARB_RATIO=12
SIMGLUCOSE_CORRECTION_FACTOR=40
SIMGLUCOSE_PRE_BOLUS_MINUTES=15
```

---

## Parameter Relationships

### The 500/1800 Rules

These approximations relate CR and CF to Total Daily Dose:

```
CR ≈ 500 / TDD
CF ≈ 1800 / TDD
```

**Implications:**
- CR and CF should be inversely related to insulin needs
- If TDD increases (more resistant), both CR and CF decrease
- If TDD decreases (more sensitive), both CR and CF increase

### Basal-to-Bolus Ratio

Typical insulin distribution:
- **Basal**: 40-50% of TDD
- **Bolus**: 50-60% of TDD

**Example for TDD = 50 units:**
- Basal: 20-25 units/day → 0.8-1.0 U/hr
- Bolus: 25-30 units/day → divided among meals

### Parameter Interdependencies

| Change | Effect on Glucose | Compensation Strategy |
|--------|-------------------|----------------------|
| ↑ Basal | ↓ Fasting glucose | May need ↓ CR to prevent meal hypos |
| ↓ Basal | ↑ Fasting glucose | May need ↓ CR to compensate |
| ↓ CR | ↓ Post-meal glucose | May need ↑ basal to prevent overnight lows |
| ↑ CR | ↑ Post-meal glucose | May need ↓ basal to prevent overnight highs |
| ↓ Target | More aggressive corrections | Increases hypo risk at meals |
| ↑ Pre-bolus | Better post-meal control | Must ensure meal timing is reliable |

---

## References

1. **UVA/Padova Simulator**: Dalla Man C, et al. "The UVA/PADOVA Type 1 Diabetes Simulator." *Journal of Diabetes Science and Technology*, 2014.

2. **simglucose Library**: https://github.com/jxx123/simglucose

3. **Insulin Dosing Guidelines**: American Diabetes Association. "Standards of Medical Care in Diabetes." *Diabetes Care*, 2024.

4. **Time in Range Consensus**: Battelino T, et al. "Clinical Targets for Continuous Glucose Monitoring Data Interpretation." *Diabetes Care*, 2019.
