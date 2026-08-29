from datetime import datetime

from parksmart import ParkSmartOptimizer, UserRequest


def test_returns_ranked_recommendations():
    optimizer = ParkSmartOptimizer(seed=42)
    request = UserRequest(
        user_id="test",
        current_x=100,
        current_y=200,
        destination_x=500,
        destination_y=600,
        max_walking_distance_m=650,
        price_sensitivity=0.3,
        requested_at=datetime(2026, 8, 29, 13),
    )
    result = optimizer.optimize_parking_recommendation(request, top_k=3)
    assert result["optimization_successful"] is True
    assert len(result["recommendations"]) == 3
    scores = [r["optimization_score"] for r in result["recommendations"]]
    assert scores == sorted(scores)


def test_walking_constraint_is_in_meters():
    optimizer = ParkSmartOptimizer(seed=42)
    request = UserRequest(
        user_id="test",
        current_x=0,
        current_y=0,
        destination_x=450,
        destination_y=450,
        max_walking_distance_m=75,
        price_sensitivity=0.5,
        requested_at=datetime(2026, 8, 29, 13),
    )
    result = optimizer.optimize_parking_recommendation(request)
    assert result["optimization_successful"] is True
    assert all(r["walking_distance_m"] <= 75 for r in result["recommendations"])


def test_time_preference_changes_objective():
    optimizer = ParkSmartOptimizer(seed=42)
    common = dict(
        user_id="test",
        current_x=0,
        current_y=0,
        destination_x=900,
        destination_y=900,
        max_walking_distance_m=1300,
        price_sensitivity=0.0,
        requested_at=datetime(2026, 8, 29, 13),
    )
    walking = optimizer.optimize_parking_recommendation(UserRequest(**common, time_preference="minimize_walking"), top_k=1)
    driving = optimizer.optimize_parking_recommendation(UserRequest(**common, time_preference="minimize_driving"), top_k=1)
    assert walking["recommendations"][0]["block_id"] != driving["recommendations"][0]["block_id"]
