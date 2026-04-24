from __future__ import annotations

import math
from dataclasses import dataclass

from app.constants import canonical_team_name
from app.schemas import MatchFeatures
from app.services.historical_data import FEATURE_ORDER
from app.services.modeling import get_or_train_model_bundle


@dataclass
class PredictionDetails:
    probability: float
    model_name: str
    data_source: str
    feature_values: dict[str, float]
    top_factors: list[str]


def _fallback_feature_values(features: MatchFeatures) -> dict[str, float]:
    team = canonical_team_name(features.team)
    opponent = canonical_team_name(features.opponent)
    toss_winner = canonical_team_name(features.toss_winner)
    toss_decision = features.toss_decision
    team_name_bias = 0.03 if len(team) < len(opponent) else -0.01
    toss_advantage = 0.08 if toss_winner == team and toss_decision == "field" else 0.0
    return {
        "recent_win_rate_diff": team_name_bias,
        "recent_runs_scored_diff": 0.0,
        "recent_runs_conceded_diff": 0.0,
        "recent_run_rate_diff": 0.0,
        "recent_wickets_taken_diff": 0.0,
        "recent_wickets_lost_diff": 0.0,
        "head_to_head_win_rate": 0.5,
        "venue_team_win_rate": 0.5,
        "venue_opponent_win_rate": 0.5,
        "venue_first_innings_avg": 170.0,
        "venue_chase_win_rate": 0.5,
        "rest_days_diff": 0.0,
        "batting_strength_diff": 0.0,
        "bowling_strength_diff": 0.0,
        "lineup_experience_diff": 0.0,
        "batter_bowler_matchup_diff": 0.0,
        "toss_won_by_team": float(toss_winner == team),
        "toss_won_by_opponent": float(toss_winner == opponent),
        "toss_decision_bat": float(toss_decision == "bat"),
        "toss_decision_field": float(toss_decision == "field"),
        "toss_advantage_team": toss_advantage,
        "known_team_players": float(len(features.team_players)),
        "known_opponent_players": float(len(features.opponent_players)),
    }


def _heuristic_probability(feature_values: dict[str, float]) -> float:
    logit = (
        0.9 * feature_values["recent_win_rate_diff"]
        + 0.002 * feature_values["recent_runs_scored_diff"]
        + 0.002 * feature_values["recent_runs_conceded_diff"]
        + 0.25 * feature_values["recent_run_rate_diff"]
        + 0.08 * feature_values["head_to_head_win_rate"]
        + 0.015 * feature_values["batting_strength_diff"]
        + 0.015 * feature_values["bowling_strength_diff"]
        + 0.06 * feature_values["batter_bowler_matchup_diff"]
        + 0.08 * feature_values["toss_advantage_team"]
    )
    probability = 1.0 / (1.0 + math.exp(-logit))
    return max(0.05, min(0.95, probability))


def _factor_text(feature_name: str, value: float) -> str | None:
    if feature_name == "recent_win_rate_diff":
        return "Current form is stronger than the opponent." if value > 0 else "Current form trails the opponent."
    if feature_name == "head_to_head_win_rate":
        return "Head-to-head history leans this way." if value > 0.55 else None
    if feature_name == "batting_strength_diff":
        return "Projected batting lineup looks stronger." if value > 0 else "Projected batting depth looks weaker."
    if feature_name == "bowling_strength_diff":
        return "Projected bowling attack looks stronger." if value > 0 else "Projected bowling attack looks weaker."
    if feature_name == "batter_bowler_matchup_diff":
        return "Historical batter-vs-bowler matchups are favorable." if value > 0 else "Historical player matchups are unfavorable."
    if feature_name == "venue_team_win_rate":
        return "Venue history is supportive." if value > 0.55 else None
    if feature_name == "toss_advantage_team":
        return "Toss and venue tendencies improved the setup." if value > 0 else None
    return None


def _top_factors(feature_values: dict[str, float]) -> list[str]:
    ranked = sorted(
        FEATURE_ORDER,
        key=lambda name: abs(feature_values.get(name, 0.0)),
        reverse=True,
    )
    factors: list[str] = []
    for name in ranked:
        text = _factor_text(name, feature_values.get(name, 0.0))
        if text and text not in factors:
            factors.append(text)
        if len(factors) == 4:
            break
    return factors or ["The model sees this matchup as close with limited separation."]


def predict_match_details(features: MatchFeatures) -> PredictionDetails:
    bundle = get_or_train_model_bundle()
    if bundle is not None:
        feature_values = bundle.repository.build_feature_dict(features)
        probability = bundle.predict_proba(feature_values)
        return PredictionDetails(
            probability=probability,
            model_name=bundle.model_name,
            data_source="trained_ipl_history",
            feature_values=feature_values,
            top_factors=_top_factors(feature_values),
        )

    feature_values = _fallback_feature_values(features)
    return PredictionDetails(
        probability=_heuristic_probability(feature_values),
        model_name="heuristic_fallback",
        data_source="fallback_without_trained_model",
        feature_values=feature_values,
        top_factors=_top_factors(feature_values),
    )


def predict_match_probability(features: MatchFeatures) -> float:
    return predict_match_details(features).probability
