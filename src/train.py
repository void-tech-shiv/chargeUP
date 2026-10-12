"""Compatibility training entry point for ChargeUp Palo Alto v2.

`python -m src.train` now runs the canonical real-data SLM pipeline rather than
the retired synthetic training path. Extra CLI flags are handled by
`src.run_palo_alto_pipeline`.
"""
from .run_palo_alto_pipeline import main

if __name__=='__main__':
    main()
