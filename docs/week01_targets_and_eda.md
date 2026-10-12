# ChargeUp — SLM Week 1: Problem Definition & Initial EDA

This file follows the ChargeUp Week 1 task from the course SLM.

## 1. Problem definition

ChargeUp has two primary machine-learning targets.

### Target A — Station fault flag (classification)

**Target column:** `station_fault_flag`

Goal: predict whether a charging-station observation indicates a fault condition.

- `0` = normal
- `1` = fault

This is a binary classification task.

### Target B — Next-hour charging demand (regression)

**Target column:** `next_hour_demand_kwh`

Goal: forecast the charging demand, in kWh, for the next hour.

This is a supervised regression task.

## 2. Input groups required by the SLM

The Week 1 exploratory analysis is organised around the three input groups called out in the ChargeUp brief:

### Charging / station usage data
- `session_count`
- `avg_session_minutes`
- `avg_power_kw`
- `connector_errors`
- `voltage_v`
- `current_a`
- `availability_pct`
- `energy_kwh`

### Time data
- `timestamp`

### Weather data
- `temperature_c`
- `humidity_pct`
- `rainfall_mm`

### Geography
- `station_id`
- `latitude`
- `longitude`

## 3. Dataset snapshot used for the initial EDA

The repository currently contains a synthetic/demo ChargeUp dataset so that the complete project can be reproduced without a proprietary charging-station source.

- Rows: **5,360**
- Charging stations: **16**
- Time range: **2026-09-01 00:00:00 to 2026-09-14 22:00:00**
- Total missing values in the generated dataset: **0**

## 4. Initial EDA — demand target

For `next_hour_demand_kwh`:

- Mean: **22.02 kWh**
- Median: **20.25 kWh**
- Standard deviation: **10.06 kWh**
- Minimum: **0.20 kWh**
- Maximum: **59.21 kWh**

### Early relationships seen in the demo data

Correlation with next-hour demand:

| Feature | Correlation |
|---|---:|
| `energy_kwh` | 0.830 |
| `session_count` | 0.599 |
| `temperature_c` | 0.385 |
| `rainfall_mm` | 0.033 |
| `availability_pct` | 0.009 |
| `connector_errors` | 0.002 |
| `humidity_pct` | -0.180 |

This is **exploratory only**. It is not model selection and does not replace later cross-validation.

### Time pattern

The highest mean next-hour demand in the current demo data occurs around the late afternoon / evening. The top observed hours are:

- 17:00
- 18:00
- 16:00
- 08:00
- 15:00

This confirms that time-derived features should be considered in later weeks.

## 5. Initial EDA — fault target

Observed fault rate:

- **3.94%**

This immediately indicates an imbalanced classification problem, which will matter later when the SLM reaches imbalance-aware evaluation.

In the current demo data:

- Normal observations have mean connector errors of about **0.17**
- Fault observations have mean connector errors of about **3.16**
- Normal observations have mean availability of about **98.98%**
- Fault observations have mean availability of about **85.33%**

These are exploratory observations only and are not treated as final causal conclusions.

## 6. Geography check

Current generated station coordinates span approximately:

- Latitude: **16.415 to 16.574**
- Longitude: **80.530 to 80.710**

The station ID plus latitude/longitude fields provide the geography inputs required for later station-level analysis and clustering.

## 7. Week 1 completion checklist

- [x] Define the station-fault classification target.
- [x] Define the next-hour charging-demand regression target.
- [x] Identify charging / station usage inputs.
- [x] Identify time inputs.
- [x] Identify weather inputs.
- [x] Identify geography inputs.
- [x] Perform an initial EDA on both targets.
- [x] Record early class-imbalance, demand, time, weather, and geography observations.

## 8. Important note

Week 1 is only for problem definition and initial EDA. Data cleaning, leakage-safe preparation, model fitting, regularisation, tree models, boosting, clustering, calibration, deployment, drift monitoring, and retraining belong to later SLM weeks and should not be counted as Week 1 work.
