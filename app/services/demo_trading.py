from __future__ import annotations

import math
from datetime import datetime

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import DemoTrade, SignalRecord, UserAccount, WatchlistSubscription
from app.schemas import (
    DemoTradeResponse,
    DemoWalletResponse,
    LiveMatchState,
    MatchFeatures,
    OddsSnapshot,
    PredictionResponse,
    SignalResponse,
    WatchRequest,
)
from app.services.notifications import send_telegram_text
from app.services.prediction import predict_match_details
from app.services.value_engine import make_value_decision


settings = get_settings()


def add_watchlist_subscription(db: Session, payload: WatchRequest) -> WatchlistSubscription:
    existing = db.scalar(
        select(WatchlistSubscription).where(
            WatchlistSubscription.telegram_user_id == payload.telegram_user_id,
            WatchlistSubscription.match_id == payload.match_id,
            WatchlistSubscription.team == payload.team,
        )
    )
    if existing is not None:
        existing.enabled = True
        existing.auto_trade = payload.auto_trade
        existing.notes = payload.notes
        existing.username = payload.username
        db.commit()
        db.refresh(existing)
        return existing

    record = WatchlistSubscription(
        telegram_user_id=payload.telegram_user_id,
        username=payload.username,
        match_id=payload.match_id,
        team=payload.team,
        opponent=payload.opponent,
        auto_trade=payload.auto_trade,
        notes=payload.notes,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def disable_watchlist_subscription(
    db: Session,
    telegram_user_id: int,
    match_id: str,
    team: str,
) -> bool:
    record = db.scalar(
        select(WatchlistSubscription).where(
            WatchlistSubscription.telegram_user_id == telegram_user_id,
            WatchlistSubscription.match_id == match_id,
            WatchlistSubscription.team == team,
            WatchlistSubscription.enabled.is_(True),
        )
    )
    if record is None:
        return False
    record.enabled = False
    db.commit()
    return True


def list_watchlist_subscriptions(
    db: Session,
    telegram_user_id: int,
) -> list[WatchlistSubscription]:
    return db.scalars(
        select(WatchlistSubscription)
        .where(
            WatchlistSubscription.telegram_user_id == telegram_user_id,
            WatchlistSubscription.enabled.is_(True),
        )
        .order_by(WatchlistSubscription.created_at.desc())
    ).all()


def list_recent_signals(db: Session, telegram_user_id: int, limit: int = 10) -> list[SignalRecord]:
    return db.scalars(
        select(SignalRecord)
        .where(SignalRecord.telegram_user_id == telegram_user_id)
        .order_by(desc(SignalRecord.created_at))
        .limit(limit)
    ).all()


def list_demo_trades(
    db: Session,
    telegram_user_id: int,
    status: str | None = None,
) -> list[DemoTrade]:
    query = select(DemoTrade).where(DemoTrade.telegram_user_id == telegram_user_id)
    if status:
        query = query.where(DemoTrade.status == status)
    return db.scalars(query.order_by(desc(DemoTrade.opened_at))).all()


def get_demo_wallet_summary(db: Session, telegram_user_id: int) -> DemoWalletResponse:
    user = db.scalar(select(UserAccount).where(UserAccount.telegram_user_id == telegram_user_id))
    if user is None:
        user = UserAccount(telegram_user_id=telegram_user_id, bankroll=settings.default_bankroll)
        db.add(user)
        db.commit()
        db.refresh(user)

    trades = list_demo_trades(db, telegram_user_id)
    open_trades = [trade for trade in trades if trade.status == "open"]
    closed_trades = [trade for trade in trades if trade.status == "closed"]
    realized_profit = sum(trade.pnl or 0.0 for trade in closed_trades)
    open_exposure = sum(trade.stake for trade in open_trades)
    available_balance = user.bankroll + realized_profit - open_exposure
    total_staked = sum(trade.stake for trade in closed_trades)
    roi = (realized_profit / total_staked) if total_staked else 0.0
    return DemoWalletResponse(
        telegram_user_id=telegram_user_id,
        starting_bankroll=user.bankroll,
        realized_profit=round(realized_profit, 2),
        open_exposure=round(open_exposure, 2),
        available_balance=round(available_balance, 2),
        open_trades=len(open_trades),
        closed_trades=len(closed_trades),
        roi=round(roi, 4),
    )


def _normalize_over_value(overs: float | None) -> float:
    if overs is None:
        return 0.0
    whole = int(overs)
    balls_part = int(round((overs - whole) * 10))
    return whole + (balls_part / 6.0)


def _live_probability_adjustment(base_probability: float, live_state: LiveMatchState, team: str) -> float:
    batting_team = live_state.batting_team or ""
    if live_state.target_runs and live_state.runs_needed is not None and live_state.balls_remaining:
        wickets_in_hand = max(0, 10 - (live_state.current_wickets or 0))
        required_rate = live_state.required_run_rate or 0.0
        current_rate = live_state.current_run_rate or 0.0
        pressure = (required_rate - current_rate) * 0.045
        wicket_bonus = (wickets_in_hand - 4.5) * 0.03
        balls_factor = min(1.0, live_state.balls_remaining / 120.0)
        chasing_probability = 0.5 - pressure + wicket_bonus * balls_factor
        chasing_probability = max(0.03, min(0.97, chasing_probability))
        if batting_team == team:
            return 0.55 * base_probability + 0.45 * chasing_probability
        return 0.55 * base_probability + 0.45 * (1.0 - chasing_probability)

    if batting_team == team and live_state.current_runs is not None and live_state.current_overs:
        overs = _normalize_over_value(live_state.current_overs)
        projected_total = live_state.current_runs / max(overs, 0.1) * 20.0
        score_push = (projected_total - 170.0) / 100.0
        return max(0.03, min(0.97, base_probability + score_push * 0.25))

    return base_probability


def build_live_match_features(
    live_state: LiveMatchState,
    team: str,
    opponent: str,
    odds: float,
) -> MatchFeatures:
    team_players = live_state.team_players if live_state.batting_team == team else live_state.opponent_players
    opponent_players = (
        live_state.opponent_players if live_state.batting_team == team else live_state.team_players
    )
    return MatchFeatures(
        team=team,
        opponent=opponent,
        odds=odds,
        venue=live_state.venue,
        match_date=None,
        team_players=team_players,
        opponent_players=opponent_players,
        notes=live_state.status,
    )


def evaluate_live_signal(
    bankroll: float,
    live_state: LiveMatchState,
    odds_snapshot: OddsSnapshot,
    team: str,
    opponent: str,
) -> PredictionResponse:
    odds = odds_snapshot.best_back_odds or odds_snapshot.last_traded_price or odds_snapshot.best_lay_odds
    if odds is None:
        raise ValueError("No usable odds were returned for the selection.")

    features = build_live_match_features(live_state, team, opponent, odds)
    details = predict_match_details(features)
    adjusted_probability = _live_probability_adjustment(details.probability, live_state, team)
    decision = make_value_decision(
        probability=adjusted_probability,
        odds=odds,
        bankroll=bankroll,
        min_edge=settings.min_edge,
        max_kelly_fraction=settings.max_kelly_fraction,
    )
    top_factors = list(details.top_factors)
    if live_state.required_run_rate is not None:
        top_factors.insert(
            0,
            f"Required rate {live_state.required_run_rate:.2f} with "
            f"{max(0, 10 - (live_state.current_wickets or 0))} wickets in hand.",
        )
    return PredictionResponse(
        team=team,
        opponent=opponent,
        win_probability=round(adjusted_probability, 4),
        implied_probability=round(decision.implied_probability, 4),
        value_edge=round(decision.value_edge, 4),
        kelly_fraction=round(decision.kelly_fraction, 4),
        suggested_stake=decision.suggested_stake,
        action=decision.action,
        exit_take_profit_odds=decision.exit_take_profit_odds,
        exit_stop_loss_odds=decision.exit_stop_loss_odds,
        explanation=decision.explanation,
        model_name=details.model_name,
        data_source=f"{details.data_source}+live_adjustment",
        top_factors=top_factors[:5],
        feature_values={key: round(value, 4) for key, value in details.feature_values.items()},
    )


def _signal_message(
    live_state: LiveMatchState,
    odds_snapshot: OddsSnapshot,
    prediction: PredictionResponse,
) -> str:
    lines = [
        f"{prediction.action.upper()} | {prediction.team} vs {prediction.opponent}",
        f"Odds: {odds_snapshot.best_back_odds or odds_snapshot.last_traded_price or 0:.2f}",
        f"Win probability: {prediction.win_probability:.2%}",
        f"Value edge: {prediction.value_edge:.2%}",
    ]
    if live_state.current_runs is not None and live_state.current_wickets is not None:
        lines.append(
            f"Live state: {live_state.batting_team or 'Batting side'} "
            f"{live_state.current_runs}/{live_state.current_wickets} "
            f"({live_state.current_overs or 0} ov)"
        )
    lines.append(f"Why: {prediction.explanation}")
    if prediction.top_factors:
        lines.append(f"Signals: {' | '.join(prediction.top_factors[:3])}")
    return "\n".join(lines)


def _create_signal_record(
    db: Session,
    telegram_user_id: int,
    match_id: str,
    team: str,
    opponent: str,
    prediction: PredictionResponse,
    message: str,
    odds: float,
) -> tuple[SignalRecord, bool]:
    latest = db.scalar(
        select(SignalRecord)
        .where(
            SignalRecord.telegram_user_id == telegram_user_id,
            SignalRecord.match_id == match_id,
            SignalRecord.team == team,
        )
        .order_by(desc(SignalRecord.created_at))
    )
    if (
        latest is not None
        and latest.signal_type == prediction.action
        and abs(latest.odds - odds) < 0.01
        and abs(latest.value_edge - prediction.value_edge) < 0.01
    ):
        return latest, False

    signal = SignalRecord(
        telegram_user_id=telegram_user_id,
        match_id=match_id,
        team=team,
        opponent=opponent,
        signal_type=prediction.action,
        odds=odds,
        probability=prediction.win_probability,
        value_edge=prediction.value_edge,
        message=message,
    )
    db.add(signal)
    db.commit()
    db.refresh(signal)
    return signal, True


def _open_trade_for_signal(
    db: Session,
    user: UserAccount,
    signal: SignalRecord,
    odds_snapshot: OddsSnapshot,
    prediction: PredictionResponse,
    match_id: str,
    notes: str | None = None,
) -> DemoTrade | None:
    existing = db.scalar(
        select(DemoTrade).where(
            DemoTrade.telegram_user_id == user.telegram_user_id,
            DemoTrade.match_id == match_id,
            DemoTrade.team == prediction.team,
            DemoTrade.status == "open",
        )
    )
    if existing is not None:
        return existing

    wallet = get_demo_wallet_summary(db, user.telegram_user_id)
    stake = min(prediction.suggested_stake, wallet.available_balance)
    if stake <= 0:
        return None
    trade = DemoTrade(
        telegram_user_id=user.telegram_user_id,
        username=user.username,
        match_id=match_id,
        market_source=odds_snapshot.provider,
        team=prediction.team,
        opponent=prediction.opponent,
        entry_odds=odds_snapshot.best_back_odds or odds_snapshot.last_traded_price or 0.0,
        current_odds=odds_snapshot.best_back_odds or odds_snapshot.last_traded_price or 0.0,
        stake=round(stake, 2),
        probability_at_entry=prediction.win_probability,
        value_edge_at_entry=prediction.value_edge,
        exit_take_profit_odds=prediction.exit_take_profit_odds,
        exit_stop_loss_odds=prediction.exit_stop_loss_odds,
        signal_id=signal.id,
        notes=notes,
    )
    db.add(trade)
    db.commit()
    db.refresh(trade)
    return trade


def _close_trade(
    db: Session,
    trade: DemoTrade,
    exit_odds: float,
    reason: str,
) -> DemoTrade:
    trade.exit_odds = exit_odds
    trade.current_odds = exit_odds
    trade.status = "closed"
    trade.closed_at = datetime.utcnow()
    trade.pnl = round(((trade.entry_odds / exit_odds) - 1.0) * trade.stake, 2)
    trade.notes = f"{trade.notes or ''}\nExit reason: {reason}".strip()
    db.commit()
    db.refresh(trade)
    return trade


def run_watchlist_scan(
    db: Session,
    subscription: WatchlistSubscription,
    live_state: LiveMatchState,
    odds_snapshot: OddsSnapshot,
    notify: bool = True,
) -> tuple[SignalRecord, DemoTrade | None]:
    user = db.scalar(select(UserAccount).where(UserAccount.telegram_user_id == subscription.telegram_user_id))
    if user is None:
        user = UserAccount(
            telegram_user_id=subscription.telegram_user_id,
            username=subscription.username,
            bankroll=settings.default_bankroll,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    wallet = get_demo_wallet_summary(db, subscription.telegram_user_id)
    prediction = evaluate_live_signal(
        bankroll=wallet.available_balance,
        live_state=live_state,
        odds_snapshot=odds_snapshot,
        team=subscription.team,
        opponent=subscription.opponent,
    )
    market_odds = odds_snapshot.best_back_odds or odds_snapshot.last_traded_price or 0.0
    message = _signal_message(live_state, odds_snapshot, prediction)
    signal, is_new_signal = _create_signal_record(
        db,
        subscription.telegram_user_id,
        subscription.match_id,
        subscription.team,
        subscription.opponent,
        prediction,
        message,
        market_odds,
    )

    open_trade = db.scalar(
        select(DemoTrade).where(
            DemoTrade.telegram_user_id == subscription.telegram_user_id,
            DemoTrade.match_id == subscription.match_id,
            DemoTrade.team == subscription.team,
            DemoTrade.status == "open",
        )
    )

    trade: DemoTrade | None = None
    if open_trade is not None:
        open_trade.current_odds = market_odds
        should_exit = (
            market_odds <= open_trade.exit_take_profit_odds
            or market_odds >= open_trade.exit_stop_loss_odds
            or prediction.action == "skip"
        )
        db.commit()
        if should_exit:
            trade = _close_trade(db, open_trade, market_odds, prediction.action)
            signal.message = f"{signal.message}\nDemo trade closed with P&L {trade.pnl:.2f}"
            db.commit()
    elif subscription.auto_trade and prediction.action == "enter":
        trade = _open_trade_for_signal(
            db,
            user,
            signal,
            odds_snapshot,
            prediction,
            subscription.match_id,
            subscription.notes,
        )
        if trade is not None:
            signal.message = f"{signal.message}\nDemo trade opened with stake {trade.stake:.2f}"
            db.commit()

    if notify and (is_new_signal or trade is not None):
        send_telegram_text(subscription.telegram_user_id, signal.message)
    return signal, trade
