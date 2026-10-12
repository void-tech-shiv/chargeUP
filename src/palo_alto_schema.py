"""Canonical schema for the real Palo Alto ChargePoint dataset pipeline.

The City of Palo Alto source does not contain weather or hardware telemetry
(voltage/current) columns. The project therefore uses only source-derived
charging, time, and geography features instead of fabricating those fields.
"""
from __future__ import annotations

SOURCE_REQUIRED_COLUMNS = [
    "Station Name",
    "Start Date",
    "End Date",
    "Total Duration (hh:mm:ss)",
    "Charging Time (hh:mm:ss)",
    "Energy (kWh)",
    "Port Type",
    "Port Number",
    "Latitude",
    "Longitude",
    "Fee",
    "Ended By",
]

BASE_FEATURES = [
    "latitude",
    "longitude",
    "session_count",
    "avg_session_minutes",
    "avg_total_duration_minutes",
    "avg_power_kw",
    "energy_kwh",
    "fee_usd",
    "level1_share",
    "level2_share",
    "dc_fast_share",
]

TIME_FEATURES = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "is_peak_hour",
]

MODEL_FEATURES = BASE_FEATURES + TIME_FEATURES + ["station_segment"]

CLUSTER_FEATURES = [
    "energy_kwh",
    "session_count",
    "avg_session_minutes",
    "avg_total_duration_minutes",
    "avg_power_kw",
    "level1_share",
    "level2_share",
    "dc_fast_share",
]

REGRESSION_TARGET = "next_hour_demand_kwh"
CLASSIFICATION_TARGET = "station_fault_flag"

# Important: the source file does not contain a true hardware-failure label.
# We use a transparent weak-supervision proxy: whether the next exact hour at
# the same station contains a session whose `Ended By` value includes "Server"
# (typically "CPS Server" in the ChargePoint export). This is intentionally
# documented as a server-termination proxy, not as ground-truth broken hardware.
FAULT_PROXY_DESCRIPTION = (
    "next-hour ChargePoint server-termination proxy derived from the source "
    "`Ended By` field; it is not a ground-truth hardware failure label"
)
