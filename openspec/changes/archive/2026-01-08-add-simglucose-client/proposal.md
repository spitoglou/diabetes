# Add Simglucose Client

## Why

The current streaming clients (`serve client` and `serve synced-client`) replay pre-recorded data from the Ohio dataset. This limits testing scenarios to historical patterns and provides finite data that eventually exhausts or repeats.

A simglucose-based client would provide:
- **Infinite synthetic data** - Continuous simulation without dataset exhaustion
- **Controllable scenarios** - Define custom meal schedules with specific times and carb amounts
- **Multiple virtual patients** - 30 FDA-approved UVA/Padova model patients (10 adults, 10 adolescents, 10 children)
- **Time-synced simulation** - Fast-forward to current time of day, then stream in real-time
- **Reproducibility** - Seed the simulation for deterministic testing

## What Changes

1. **Add simglucose dependency** to `pyproject.toml`
2. **Create SimglucoseProvider** class in `src/bgc_providers/` following the existing provider pattern
3. **Add `serve simglucose-client` CLI command** with options for patient, meal scenario, and time sync
4. **Add default meal scenario configuration** in settings

## Scope

- **In scope:**
  - New CLI command `serve simglucose-client`
  - SimglucoseProvider class with time-synced streaming
  - Configurable meal scenarios (default + custom)
  - Patient selection from simglucose's virtual patient pool
  - Configurable insulin modes: none, basal, basal-bolus
  - Verbose mode for debugging

- **Out of scope:**
  - Real-time meal injection via API
  - Integration with the mobile dashboard
  - Gymnasium/RL training integration

## Risks

- **Low:** Simglucose is a well-maintained library used in diabetes research
- **Medium:** Time sync requires fast-forwarding simulation which may take a few seconds on startup
