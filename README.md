# Urban Parking MILP Optimizer

A reproducible synthetic urban parking recommendation system using binary mixed-integer linear programming (MILP) to rank parking blocks by travel time, walking distance, availability, price, safety, restrictions, and user preferences.

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

## Tests

Run the test suite locally with:

```bash
pip install pytest
pytest -q
```

GitHub Actions runs the tests automatically on pushes and pull requests targeting `main` across Python 3.10, 3.11, and 3.12.

## Suggested repository topics

`operations-research` · `milp` · `mixed-integer-programming` · `optimization` · `parking` · `urban-mobility` · `python` · `scipy` · `decision-support`

## License

This repository is source-available for strictly non-commercial use only.

**Commercial use is prohibited without prior express written permission from the copyright holder.** This includes use in commercial products, paid services, SaaS offerings, client work, consulting engagements, internal business systems, revenue-generating workflows, and other for-profit activities.

See [`LICENSE`](LICENSE) for the complete terms.

## Scope

This is a synthetic optimization example, not a production parking service. Business-impact figures, real-time sensor integration, Bayesian inference, and ML demand prediction are intentionally not claimed unless actually implemented.
