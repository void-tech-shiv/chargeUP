# ChargeUp — Week 2 Completion (Palo Alto real data)

Week 2 is complete through `src/palo_alto_data.py`.

- Current City download endpoint + reproducible fallback mirror
- Real source schema validation
- Timestamp and duration parsing
- Numeric energy/fee/geography validation
- Source-invalid row handling
- Station-hour aggregation
- Exact `t + 1 hour` demand target construction
- Next-hour CPS/server-termination weak label
- Leakage guard: target-generating audit columns excluded from model features
- Generated full cleaned table, small real-data sample and data-quality report

Authoritative generated outputs:

- `data/processed/chargeup_sessions_clean.csv` (generated; Git-ignored)
- `data/processed/palo_alto_hourly_sample.csv`
- `reports/palo_alto_data_quality.json`
