# ChargeUp — SLM Week 1: Real Dataset, Targets & Initial EDA

Week 1 now uses the **City of Palo Alto Electric Vehicle Charging Station Usage (July 2011–Dec 2020)** source instead of the earlier synthetic demo data.

Official dataset page: https://data.paloalto.gov/datasets/194693/electric-vehicle-charging-station-usage-july-2011-dec-2020/

Source file: `ChargePoint Data CY20Q4.csv`

## 1. Source fields used

The City ChargePoint CSV provides real charging-session, time and geography fields including:

- `Station Name`
- `Start Date`, `End Date`
- `Total Duration (hh:mm:ss)`
- `Charging Time (hh:mm:ss)`
- `Energy (kWh)`
- `Port Type`, `Port Number`, `Plug Type`
- `Latitude`, `Longitude`
- `Fee`
- `Ended By`

It does **not** provide weather, voltage, current, or a ground-truth broken-station label. Those fields are not fabricated.

## 2. Regression target

**Column:** `next_hour_demand_kwh`

For every station-hour observation, the target is the energy delivered at the **same station in the next exact clock hour**. If the station has no charging session in that next hour, the target is `0` kWh.

This avoids the common leakage/error of shifting to the next *observed* row even when it occurs many hours later.

## 3. Classification target

**Column:** `station_fault_flag`

The source has no verified hardware-failure field. To keep the SLM classification task tied to the real dataset, ChargeUp uses a transparent weak-supervision proxy:

- `1` = the next exact hour at the same station contains at least one session whose source `Ended By` value includes `Server` (for example `CPS Server`)
- `0` = no such server-ended session in the next exact hour

This is a **next-hour server-termination risk proxy**, not proof that a charging station is physically broken.

The source `Ended By` field and any derived current/future server-termination count are excluded from model inputs so the label is not leaked into the features.

## 4. Input groups used by the real-data model

### Charging / usage
- `session_count`
- `avg_session_minutes`
- `avg_total_duration_minutes`
- `avg_power_kw`
- `energy_kwh`
- `fee_usd`
- `level1_share`
- `level2_share`
- `dc_fast_share`

### Time
- station-hour `timestamp`
- later cyclic hour/day-of-week features

### Geography
- `station_id`
- `latitude`
- `longitude`

### Weather
The supplied City dataset has no weather columns. Weather is therefore documented as unavailable in this source rather than filled with synthetic values.

## 5. Initial EDA workflow

`python -m src.palo_alto_data --download` produces:

- `data/processed/chargeup_sessions_clean.csv`
- `data/processed/palo_alto_hourly_sample.csv`
- `reports/palo_alto_data_quality.json`

The quality report records the actual downloaded row count, valid-session count, station count, time range, server-ended-session count and resulting classification positive rate. These generated values are the authoritative Week 1/2 dataset statistics.

## 6. Week 1 checklist

- [x] Use the instructor-linked Palo Alto dataset family.
- [x] Identify real source charging/time/geography columns.
- [x] Define next-exact-hour demand regression target.
- [x] Define a transparent server-termination proxy for the missing fault label.
- [x] Document source limitations instead of fabricating absent telemetry/weather.
- [x] Keep target-generating `Ended By` information out of model features.
