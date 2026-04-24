from dataclasses import dataclass


@dataclass
class ValueDecision:
    implied_probability: float
    value_edge: float
    kelly_fraction: float
    suggested_stake: float
    action: str
    exit_take_profit_odds: float
    exit_stop_loss_odds: float
    explanation: str


def calculate_value(probability: float, odds: float) -> float:
    return (probability * odds) - 1.0


def calculate_kelly_fraction(probability: float, odds: float, max_fraction: float) -> float:
    b = odds - 1.0
    raw_kelly = ((probability * odds) - 1.0) / b if b > 0 else 0.0
    safe_kelly = max(0.0, raw_kelly) * 0.5
    return min(safe_kelly, max_fraction)


def suggest_exit_levels(entry_odds: float) -> tuple[float, float]:
    take_profit = max(1.01, round(entry_odds * 0.88, 2))
    stop_loss = round(entry_odds * 1.12, 2)
    return take_profit, stop_loss


def make_value_decision(
    probability: float,
    odds: float,
    bankroll: float,
    min_edge: float,
    max_kelly_fraction: float,
) -> ValueDecision:
    implied_probability = 1.0 / odds
    value_edge = calculate_value(probability, odds)
    kelly_fraction = calculate_kelly_fraction(probability, odds, max_kelly_fraction)
    suggested_stake = round(bankroll * kelly_fraction, 2)
    take_profit, stop_loss = suggest_exit_levels(odds)

    if value_edge >= min_edge and suggested_stake > 0:
        action = "enter"
        explanation = (
            "Positive value detected: model probability is above market implied probability "
            "with enough margin to justify a trade."
        )
    elif value_edge > 0:
        action = "watch"
        explanation = (
            "Slight edge detected, but the margin is below the configured safety threshold."
        )
    else:
        action = "skip"
        explanation = "No positive value edge versus the current market odds."

    return ValueDecision(
        implied_probability=implied_probability,
        value_edge=value_edge,
        kelly_fraction=kelly_fraction,
        suggested_stake=suggested_stake,
        action=action,
        exit_take_profit_odds=take_profit,
        exit_stop_loss_odds=stop_loss,
        explanation=explanation,
    )
