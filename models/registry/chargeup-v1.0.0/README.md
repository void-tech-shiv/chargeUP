# ChargeUp v1.0.0

This directory is the Week 11 versioned model registry entry.

Committed files contain the model manifest, feature schema, station-segment mapping, and PSI reference baseline. The binary `regression_model.joblib` and `fault_model.joblib` files are reproducible build artifacts and are generated with:

```bash
python -m src.week11_package_champion
```

The manifest records the expected SHA-256 checksums of those generated artifacts. Docker startup also generates the binaries before FastAPI starts when they are not already present.
