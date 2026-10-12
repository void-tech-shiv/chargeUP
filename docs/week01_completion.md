# ChargeUp — Week 1 Completion (Palo Alto real data)

Week 1 is complete on the instructor-linked City of Palo Alto dataset family.

- Real source identified: `ChargePoint Data CY20Q4.csv`
- Official current dataset page documented
- Real charging/time/geography fields identified
- Regression target: next exact-hour station energy (`next_hour_demand_kwh`)
- Classification target: next-hour server-termination proxy (`station_fault_flag`)
- Fault-label limitation documented: source has no ground-truth broken-hardware label
- Weather/voltage/current limitation documented rather than fabricated
- Target-generating `Ended By` information excluded from model features

See `docs/week01_targets_and_eda.md` and generated `reports/palo_alto_data_quality.json`.
