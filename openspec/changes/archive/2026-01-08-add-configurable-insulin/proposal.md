# Add Configurable Insulin Controller

## Why

The current simglucose-client implementation offers three insulin modes (`none`, `basal`, `basal-bolus`) but uses fixed parameters derived from patient models. Real-world diabetes management requires individualized insulin dosing, and researchers/developers need to simulate various therapy scenarios.

**Current limitations:**
- Basal rate is automatically calculated from patient steady-state parameters - no override possible
- Carb ratio (CR) and correction factor (CF) are locked to patient-specific values from simglucose's quest CSV
- Target glucose is hardcoded to 140 mg/dL in BBController
- Boluses are delivered instantly at meal time - no pre-bolus timing support
- No way to simulate suboptimal therapy adherence or parameter miscalculation scenarios

**Use cases enabled by this change:**
1. **Clinical scenario testing** - Simulate pump settings that don't match patient physiology
2. **Education** - Demonstrate how different insulin parameters affect glucose control
3. **Algorithm development** - Test prediction models against various insulin delivery patterns
4. **Research** - Reproduce specific therapy scenarios from literature

## What Changes

1. **Create `ConfigurableController`** - New controller class in `simglucose_provider.py` with overridable parameters
2. **Add CLI options** - `--basal-rate`, `--target-glucose`, `--carb-ratio`, `--correction-factor`, `--pre-bolus-minutes`
3. **Add environment variables** - Corresponding settings for all new parameters
4. **Update documentation** - Full algorithmic and clinical documentation for each parameter

## Scope

**In scope:**
- ConfigurableController class with parameter overrides
- CLI options for all configurable parameters
- Environment variable support via settings
- Comprehensive documentation (algorithmic formulas + clinical context)
- Validation of parameter ranges

**Out of scope:**
- Time-varying basal rates (basal profiles)
- Insulin-on-board (IOB) tracking
- Auto-mode / closed-loop algorithms
- Dual-wave or extended boluses

## Risks

- **Low:** Parameter validation prevents physiologically impossible values
- **Medium:** Users unfamiliar with insulin dosing may configure unrealistic scenarios - mitigated by documentation and sensible defaults

## Clinical Background

### Basal Insulin
Continuous background insulin that controls glucose between meals and overnight. Delivered at a constant rate (U/hr) to match hepatic glucose output.

### Bolus Insulin
Discrete doses to cover meals (carb coverage) and correct high glucose (correction dose).

### Key Parameters

| Parameter | Clinical Purpose | Typical Range |
|-----------|------------------|---------------|
| Basal Rate | Background glucose control | 0.5-2.0 U/hr (adults) |
| Carb Ratio (CR) | Grams of carbs covered by 1U insulin | 5-25 g/U |
| Correction Factor (CF) | BG drop per 1U insulin | 20-100 mg/dL/U |
| Target Glucose | Goal BG for corrections | 100-150 mg/dL |
| Pre-bolus Time | Minutes before meal to bolus | 0-30 min |
