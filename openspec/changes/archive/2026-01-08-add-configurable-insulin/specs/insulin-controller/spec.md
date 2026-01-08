# insulin-controller Specification

## Purpose

Defines the configurable insulin controller for the simglucose-client, enabling user-specified insulin delivery parameters with full algorithmic and clinical documentation.

## ADDED Requirements

### Requirement: Configurable Basal Rate

The ConfigurableController SHALL allow users to override the basal insulin rate.

#### Scenario: Default basal rate uses patient parameters
- **GIVEN** no `--basal-rate` option is provided
- **AND** no `SIMGLUCOSE_BASAL_RATE` environment variable is set
- **WHEN** the ConfigurableController calculates basal insulin
- **THEN** basal rate is computed as `u2ss × BW / 6000` U/min using patient-specific parameters
- **AND** the rate matches what BBController would produce for the same patient

#### Scenario: CLI override for basal rate
- **GIVEN** user runs `uv run python cli.py serve simglucose-client --basal-rate 1.5`
- **WHEN** the ConfigurableController delivers basal insulin
- **THEN** basal rate is 1.5 U/hr (0.025 U/min)
- **AND** patient-specific parameters are ignored for basal calculation

#### Scenario: Environment variable override for basal rate
- **GIVEN** `SIMGLUCOSE_BASAL_RATE=1.2` is set in environment
- **AND** no `--basal-rate` CLI option is provided
- **WHEN** the ConfigurableController delivers basal insulin
- **THEN** basal rate is 1.2 U/hr

#### Scenario: Basal rate validation
- **GIVEN** user provides `--basal-rate 10.0`
- **WHEN** the controller is initialized
- **THEN** an error is raised indicating basal rate must be between 0.0 and 5.0 U/hr

---

### Requirement: Configurable Target Glucose

The ConfigurableController SHALL allow users to specify the target blood glucose for correction boluses.

#### Scenario: Default target glucose
- **GIVEN** no `--target-glucose` option is provided
- **WHEN** the ConfigurableController calculates correction boluses
- **THEN** target glucose is 140 mg/dL (matching BBController default)

#### Scenario: CLI override for target glucose
- **GIVEN** user runs `uv run python cli.py serve simglucose-client --target-glucose 120`
- **WHEN** current glucose is 200 mg/dL
- **AND** correction factor is 40 mg/dL/U
- **THEN** correction bolus is (200 - 120) / 40 = 2.0 units

#### Scenario: Target glucose validation
- **GIVEN** user provides `--target-glucose 50`
- **WHEN** the controller is initialized
- **THEN** an error is raised indicating target glucose must be between 70 and 200 mg/dL

---

### Requirement: Configurable Carbohydrate Ratio

The ConfigurableController SHALL allow users to override the carbohydrate ratio (CR) for meal boluses.

#### Scenario: Default carb ratio uses patient parameters
- **GIVEN** no `--carb-ratio` option is provided
- **WHEN** the ConfigurableController calculates meal boluses
- **THEN** CR is loaded from simglucose's quest CSV for the specified patient
- **AND** meal bolus matches what BBController would produce

#### Scenario: CLI override for carb ratio
- **GIVEN** user runs `uv run python cli.py serve simglucose-client --carb-ratio 15`
- **WHEN** a 60g carbohydrate meal occurs
- **THEN** meal bolus is 60 / 15 = 4.0 units

#### Scenario: Carb ratio validation
- **GIVEN** user provides `--carb-ratio 0`
- **WHEN** the controller is initialized
- **THEN** an error is raised indicating carb ratio must be between 1 and 50 g/U

---

### Requirement: Configurable Correction Factor

The ConfigurableController SHALL allow users to override the correction factor (CF) for correction boluses.

#### Scenario: Default correction factor uses patient parameters
- **GIVEN** no `--correction-factor` option is provided
- **WHEN** the ConfigurableController calculates correction boluses
- **THEN** CF is loaded from simglucose's quest CSV for the specified patient
- **AND** correction bolus matches what BBController would produce

#### Scenario: CLI override for correction factor
- **GIVEN** user runs `uv run python cli.py serve simglucose-client --correction-factor 50`
- **WHEN** current glucose is 250 mg/dL
- **AND** target glucose is 140 mg/dL
- **THEN** correction bolus is (250 - 140) / 50 = 2.2 units

#### Scenario: Correction factor validation
- **GIVEN** user provides `--correction-factor 0`
- **WHEN** the controller is initialized
- **THEN** an error is raised indicating correction factor must be between 5 and 200 mg/dL/U

---

### Requirement: Configurable Pre-Bolus Time

The ConfigurableController SHALL allow users to specify how many minutes before a meal the bolus is delivered.

#### Scenario: Default pre-bolus time (no pre-bolus)
- **GIVEN** no `--pre-bolus-minutes` option is provided
- **WHEN** a meal occurs at simulation time 12:00
- **THEN** bolus is delivered at 12:00 (when meal appears in observation)
- **AND** behavior matches BBController

#### Scenario: Pre-bolus enabled
- **GIVEN** user runs `uv run python cli.py serve simglucose-client --pre-bolus-minutes 15`
- **AND** meal schedule includes a 70g meal at 12:00
- **WHEN** simulation time reaches 11:45
- **THEN** meal bolus for 70g is delivered at 11:45
- **AND** no additional bolus is delivered at 12:00 for the same meal

#### Scenario: Pre-bolus with correction at meal time
- **GIVEN** `--pre-bolus-minutes 15` is set
- **AND** meal bolus was delivered at 11:45 for a 12:00 meal
- **WHEN** simulation time reaches 12:00
- **AND** current glucose is 200 mg/dL (above correction threshold)
- **THEN** a correction bolus is delivered at 12:00
- **AND** no duplicate meal bolus is delivered

#### Scenario: Pre-bolus validation
- **GIVEN** user provides `--pre-bolus-minutes 60`
- **WHEN** the controller is initialized
- **THEN** an error is raised indicating pre-bolus time must be between 0 and 45 minutes

---

### Requirement: BBController Equivalence

The ConfigurableController with default parameters SHALL produce identical insulin actions to simglucose's BBController.

#### Scenario: Basal rate equivalence
- **GIVEN** ConfigurableController with no parameter overrides
- **AND** BBController initialized with same patient
- **WHEN** both controllers compute basal rate
- **THEN** basal rates are identical within floating-point tolerance (1e-6)

#### Scenario: Meal bolus equivalence
- **GIVEN** ConfigurableController with no parameter overrides
- **AND** BBController initialized with same patient
- **WHEN** a 45g meal occurs with glucose at 120 mg/dL
- **THEN** both controllers produce identical bolus amounts within tolerance

#### Scenario: Correction bolus equivalence
- **GIVEN** ConfigurableController with no parameter overrides
- **AND** BBController initialized with same patient
- **WHEN** glucose is 200 mg/dL with no meal
- **THEN** both controllers produce identical correction bolus within tolerance

#### Scenario: Full simulation equivalence
- **GIVEN** ConfigurableController with no parameter overrides (pre_bolus_minutes=0)
- **AND** BBController initialized with same patient
- **WHEN** both run a 24-hour simulation with identical meal schedule and seed
- **THEN** glucose traces have RMSE < 0.1 mg/dL

---

### Requirement: Verbose Output for Insulin Actions

The simglucose-client SHALL display detailed insulin information in verbose mode.

#### Scenario: Verbose output shows configurable parameters
- **GIVEN** user runs with `--verbose` and custom parameters
- **WHEN** simulation streams readings
- **THEN** output includes the configured basal rate, CR, CF, and target
- **AND** each bolus shows the calculation breakdown

#### Scenario: Pre-bolus verbose output
- **GIVEN** `--pre-bolus-minutes 15` and `--verbose` are set
- **WHEN** a pre-bolus is delivered
- **THEN** output indicates "Pre-bolus for 12:00 meal delivered at 11:45"
- **AND** shows the bolus amount and carb coverage
