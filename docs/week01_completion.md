# ChargeUp — Week 1 Completion Checklist

This file closes Week 1 of the ChargeUp project and records the work completed before moving to Week 2.

## Week 1 objective

Define the two supervised-learning targets and complete the initial exploratory analysis using charging-session, time, weather, and geography-related features.

## Targets defined

### 1. Station fault classification
- Task type: Binary classification
- Target: station fault flag
- Output: fault / no-fault prediction for a charging-station observation

### 2. Next-hour charging demand
- Task type: Regression
- Target: next-hour charging demand in kWh
- Output: estimated energy demand for the next hour

## Initial EDA completed

- Charging-session data reviewed
- Time-based variables reviewed
- Weather variables reviewed
- Geography / station-location variables reviewed
- Demand distribution inspected
- Fault-class distribution inspected
- Hourly usage patterns inspected
- Station-level differences inspected
- Dataset dimensions and station count checked

## Reproducible Week 1 files

- `docs/week01_targets_and_eda.md` — target definitions and Week 1 EDA notes
- `src/week01_eda.py` — reproducible Week 1 EDA script
- `data/raw/chargeup_sessions_sample.csv` — lightweight example of the project data schema

## Current demo-data observations

The generated demo dataset used for development contains:
- 5,360 observations
- 16 charging stations
- Approximately 3.94% fault observations

The low fault percentage indicates a class-imbalance issue that must be handled and evaluated carefully in later model-development weeks.

## Week 1 status

**COMPLETE**

No Week 2 preprocessing changes are included in this completion step.

## Next step

Proceed to Week 2 only after this checklist is accepted. Week 2 will focus on ingestion/cleaning, missing-value handling, outlier handling, leakage-safe preparation, and documented preprocessing.
