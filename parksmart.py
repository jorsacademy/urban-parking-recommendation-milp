from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from time import perf_counter
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

TimePreference = Literal["minimize_total", "minimize_walking", "minimize_driving"]


@dataclass(slots=True)
class ParkingBlock:
    block_id: str
    x_coord: float
    y_coord: float
    total_spots: int
    base_availability: float
    price_per_hour: float
    has_time_restrictions: bool
    safety_score: float


@dataclass(slots=True)
class UserRequest:
    user_id: str
    current_x: float
    current_y: float
    destination_x: float
    destination_y: float
    max_walking_distance_m: float
    price_sensitivity: float
    time_preference: TimePreference = "minimize_total"
    requested_at: datetime | None = None
    min_availability_probability: float = 0.10

    def __post_init__(self) -> None:
        if self.max_walking_distance_m < 0:
            raise ValueError("max_walking_distance_m must be non-negative")
        if not 0 <= self.price_sensitivity <= 1:
            raise ValueError("price_sensitivity must be in [0, 1]")
        if not 0 <= self.min_availability_probability <= 1:
            raise ValueError("min_availability_probability must be in [0, 1]")


class ParkSmartOptimizer:
    """Synthetic urban parking recommender backed by a binary MILP model."""

    def __init__(self, seed: int = 42, grid_size: int = 10, block_spacing_m: float = 100.0):
        self.rng = np.random.default_rng(seed)
        self.grid_size = grid_size
        self.block_spacing_m = block_spacing_m
        self.blocks: list[ParkingBlock] = []
        self.historical_data: dict[str, pd.DataFrame] = {}
        self._setup_city_data()
        self._generate_historical_data()

    def _setup_city_data(self) -> None:
        center = (self.grid_size - 1) * self.block_spacing_m / 2
        max_distance = np.hypot(center, center)

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                x = col * self.block_spacing_m
                y = row * self.block_spacing_m
                distance_from_center = np.hypot(x - center, y - center)
                normalized_distance = distance_from_center / max_distance if max_distance else 0.0

                # Downtown blocks are busier, so free-space probability is lower near the center.
                base_availability = 0.25 + 0.55 * normalized_distance
                base_availability *= self.rng.uniform(0.90, 1.10)
                base_availability = float(np.clip(base_availability, 0.10, 0.90))

                # Downtown parking is more expensive.
                price = 2.0 + 5.0 * (1.0 - normalized_distance)

                self.blocks.append(
                    ParkingBlock(
                        block_id=f"Block_{chr(65 + row)}{col + 1}",
                        x_coord=float(x),
                        y_coord=float(y),
                        total_spots=int(self.rng.integers(15, 60)),
                        base_availability=base_availability,
                        price_per_hour=round(float(price), 2),
                        has_time_restrictions=bool(self.rng.random() < 0.30),
                        safety_score=float(self.rng.uniform(6.0, 9.5)),
                    )
                )

    def _generate_historical_data(self, days: int = 30) -> None:
        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        for block in self.blocks:
            rows: list[dict[str, object]] = []
            for day_offset in range(days):
                date = now - timedelta(days=day_offset)
                for hour in range(6, 23):
                    if date.weekday() >= 5:
                        occupancy = self.rng.beta(2, 2) if 12 <= hour <= 20 else self.rng.beta(1, 4)
                    elif 9 <= hour <= 17:
                        occupancy = self.rng.beta(3, 2)
                    elif 18 <= hour <= 21:
                        occupancy = self.rng.beta(2, 2)
                    else:
                        occupancy = self.rng.beta(1, 3)

                    rows.append(
                        {
                            "datetime": date.replace(hour=hour),
                            "occupancy_rate": float(min(0.95, occupancy)),
                        }
                    )
            self.historical_data[block.block_id] = pd.DataFrame(rows)

    @staticmethod
    def _euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
        return float(np.hypot(x1 - x2, y1 - y2))

    def _historical_availability(self, block: ParkingBlock, requested_at: datetime) -> float:
        history = self.historical_data[block.block_id]
        same_hour = history[history["datetime"].dt.hour == requested_at.hour]
        same_day_type = same_hour[
            (same_hour["datetime"].dt.weekday >= 5) == (requested_at.weekday() >= 5)
        ]
        occupancy = float((same_day_type if not same_day_type.empty else same_hour)["occupancy_rate"].mean())
        historical_free = 1.0 - occupancy
        return float(np.clip(0.55 * block.base_availability + 0.45 * historical_free, 0.05, 0.95))

    @staticmethod
    def _time_weights(preference: TimePreference) -> tuple[float, float]:
        if preference == "minimize_walking":
            return 0.65, 1.65
        if preference == "minimize_driving":
            return 1.65, 0.65
        return 1.0, 1.0

    def _candidate_table(self, request: UserRequest) -> pd.DataFrame:
        requested_at = request.requested_at or datetime.now()
        driving_weight, walking_weight = self._time_weights(request.time_preference)

        records: list[dict[str, object]] = []
        for block in self.blocks:
            driving_distance_m = self._euclidean_distance(
                block.x_coord, block.y_coord, request.current_x, request.current_y
            )
            walking_distance_m = self._euclidean_distance(
                block.x_coord, block.y_coord, request.destination_x, request.destination_y
            )
            if walking_distance_m > request.max_walking_distance_m:
                continue

            # Deterministic expected travel times; no per-request random noise.
            driving_time_min = driving_distance_m / 250.0  # 15 km/h urban effective speed
            walking_time_min = walking_distance_m / 80.0   # 4.8 km/h walking speed
            availability = self._historical_availability(block, requested_at)
            if availability < request.min_availability_probability:
                continue

            expected_search_penalty = (1.0 - availability) * 6.0
            price_penalty = request.price_sensitivity * block.price_per_hour
            safety_penalty = (10.0 - block.safety_score) * 0.20
            restriction_penalty = 1.0 if block.has_time_restrictions else 0.0

            objective_score = (
                driving_weight * driving_time_min
                + walking_weight * walking_time_min
                + expected_search_penalty
                + price_penalty
                + safety_penalty
                + restriction_penalty
            )

            records.append(
                {
                    "block_id": block.block_id,
                    "coordinates": (block.x_coord, block.y_coord),
                    "driving_distance_m": driving_distance_m,
                    "walking_distance_m": walking_distance_m,
                    "estimated_driving_time_min": driving_time_min,
                    "walking_time_min": walking_time_min,
                    "availability_probability": availability,
                    "price_per_hour": block.price_per_hour,
                    "safety_score": block.safety_score,
                    "has_restrictions": block.has_time_restrictions,
                    "objective_score": objective_score,
                }
            )

        return pd.DataFrame(records)

    @staticmethod
    def _solve_one_hot_milp(scores: np.ndarray, excluded: set[int] | None = None) -> int:
        """Select exactly one candidate with binary decision variables."""
        n = len(scores)
        if n == 0:
            raise ValueError("No candidates to optimize")

        lower = np.zeros(n)
        upper = np.ones(n)
        if excluded:
            upper[list(excluded)] = 0.0

        result = milp(
            c=scores,
            integrality=np.ones(n, dtype=int),
            bounds=Bounds(lower, upper),
            constraints=LinearConstraint(np.ones((1, n)), lb=[1.0], ub=[1.0]),
            options={"disp": False},
        )
        if not result.success or result.x is None:
            raise RuntimeError(f"MILP solver failed: {result.message}")
        return int(np.argmax(result.x))

    def optimize_parking_recommendation(self, request: UserRequest, top_k: int = 3) -> dict[str, object]:
        start = perf_counter()
        candidates = self._candidate_table(request)
        if candidates.empty:
            return {
                "user_id": request.user_id,
                "optimization_successful": False,
                "error": "No parking blocks satisfy walking-distance and availability constraints.",
                "recommendations": [],
                "optimization_time_seconds": perf_counter() - start,
            }

        scores = candidates["objective_score"].to_numpy(dtype=float)
        excluded: set[int] = set()
        selected: list[int] = []

        for _ in range(min(top_k, len(candidates))):
            idx = self._solve_one_hot_milp(scores, excluded)
            selected.append(idx)
            excluded.add(idx)

        recommendations: list[dict[str, object]] = []
        for rank, idx in enumerate(selected, start=1):
            row = candidates.iloc[idx]
            recommendations.append(
                {
                    "rank": rank,
                    "block_id": row["block_id"],
                    "coordinates": row["coordinates"],
                    "availability_probability": round(float(row["availability_probability"]), 3),
                    "estimated_driving_time_min": round(float(row["estimated_driving_time_min"]), 2),
                    "walking_time_min": round(float(row["walking_time_min"]), 2),
                    "walking_distance_m": round(float(row["walking_distance_m"]), 1),
                    "price_per_hour": float(row["price_per_hour"]),
                    "safety_score": round(float(row["safety_score"]), 1),
                    "has_restrictions": bool(row["has_restrictions"]),
                    "optimization_score": round(float(row["objective_score"]), 3),
                }
            )

        return {
            "user_id": request.user_id,
            "optimization_successful": True,
            "recommendations": recommendations,
            "total_blocks_evaluated": len(candidates),
            "optimization_time_seconds": perf_counter() - start,
        }


def demo() -> None:
    optimizer = ParkSmartOptimizer(seed=42)
    request = UserRequest(
        user_id="U001_BusinessMeeting",
        current_x=100,
        current_y=200,
        destination_x=500,
        destination_y=600,
        max_walking_distance_m=650,
        price_sensitivity=0.3,
        time_preference="minimize_total",
    )
    result = optimizer.optimize_parking_recommendation(request)
    print(pd.DataFrame(result["recommendations"]).to_string(index=False))


if __name__ == "__main__":
    demo()
