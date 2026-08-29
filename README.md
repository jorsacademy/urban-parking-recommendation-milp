# Urban Parking MILP Optimizer

A reproducible synthetic parking recommendation example using a binary mixed-integer linear program (MILP).

## What it models

For a user request, the optimizer filters parking blocks by maximum walking distance and minimum availability probability, then selects the best block using a binary one-hot MILP objective that combines:

- expected driving time,
- walking time,
- parking availability/search risk,
- hourly price sensitivity,
- safety,
- time restrictions,
- user time preference.

Historical synthetic occupancy is used to adjust availability by hour and weekday/weekend pattern.

## Run

```bash
python -m pip install -r requirements.txt
python parksmart.py
```

## Notes

This is a synthetic optimization example, not a production parking service. Business-impact figures, real-time sensor integration, Bayesian inference, and ML demand prediction are intentionally not claimed unless actually implemented.
