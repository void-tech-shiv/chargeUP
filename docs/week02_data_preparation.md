# ChargeUp — Week 2: Data Preparation

This week follows the SLM data-preparation stage before model building.

## Week 2 objectives

1. Ingest the charging-session dataset in a reproducible way.
2. Validate schema and required columns before processing.
3. Detect and handle missing values.
4. Detect invalid or extreme values and document the outlier policy.
5. Parse and sort timestamps correctly.
6. Keep station/time ordering intact for the next-hour demand target.
7. Avoid data leakage by separating target construction from future-unavailable features.
8. Produce a cleaned dataset plus a compact data-quality report.

## Required checks

- Duplicate rows and duplicate session identifiers
- Missing values by column
- Invalid timestamps
- Negative charging demand / duration values
- Impossible utilization values
- Fault-label validity
- Station identifiers and geographic fields
- Weather-field ranges
- Chronological ordering

## Leakage-safe rule

For a record at time `t`, features must only contain information available at or before `t`. The regression target is the station demand for the next hour, so future demand, future fault state, or future-derived rolling statistics must not be used as features.

## Outputs planned for Week 2

- `src/week02_cleaning.py`
- cleaned data output under `data/processed/`
- data-quality summary under `reports/`
- Week 2 completion checklist

## Status

- [x] Week 2 preparation plan documented
- [ ] Cleaning script implemented
- [ ] Missing-value handling implemented
- [ ] Outlier handling implemented
- [ ] Leakage checks implemented
- [ ] Cleaned dataset/report generated
- [ ] Week 2 closed
