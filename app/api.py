from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.schemas import (
    BetResponse,
    DemoTradeResponse,
    DemoWalletResponse,
    EnterBetRequest,
    LiveMatchState,
    LiveMatchResponse,
    LiveScanRequest,
    MatchFeatures,
    PredictionResponse,
    RoiResponse,
    SignalResponse,
    WatchRequest,
    WatchResponse,
)
from app.services.betting import count_bets, enter_bet, get_or_create_user, get_roi_summary
from app.services.demo_trading import (
    add_watchlist_subscription,
    disable_watchlist_subscription,
    get_demo_wallet_summary,
    list_demo_trades,
    list_recent_signals,
    list_watchlist_subscriptions,
    run_watchlist_scan,
)
from app.services.live_data import live_data_provider
from app.services.modeling import load_model_bundle, train_model_bundle
from app.services.odds_data import betfair_odds_provider
from app.services.prediction import predict_match_details
from app.services.value_engine import make_value_decision


settings = get_settings()
router = APIRouter()


@router.get("/health")
def health() -> dict:
    bundle = load_model_bundle()
    return {
        "status": "ok",
        "app": settings.app_name,
        "model_trained": bundle is not None,
    }


@router.get("/stats")
def stats(db: Session = Depends(get_db)) -> dict:
    return {"total_bets": count_bets(db)}


@router.get("/model/status")
def model_status() -> dict:
    bundle = load_model_bundle()
    if bundle is None:
        return {"trained": False}
    return {
        "trained": True,
        "model_name": bundle.model_name,
        "trained_at": bundle.trained_at,
        "sample_count": bundle.sample_count,
        "metrics": bundle.metrics,
    }


@router.post("/model/train")
def train_model(refresh_data: bool = True) -> dict:
    bundle = train_model_bundle(refresh_data=refresh_data)
    return {
        "trained": True,
        "model_name": bundle.model_name,
        "trained_at": bundle.trained_at,
        "sample_count": bundle.sample_count,
        "metrics": bundle.metrics,
    }


@router.get("/live/matches", response_model=list[LiveMatchResponse])
def live_matches() -> list[LiveMatchResponse]:
    return live_data_provider.current_matches()


@router.get("/live/state/{match_id}", response_model=LiveMatchState)
def live_match_state(match_id: str) -> LiveMatchState:
    state = live_data_provider.match_state(match_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"No live state found for match {match_id}")
    return state


@router.post("/predict", response_model=PredictionResponse)
def predict(features: MatchFeatures, db: Session = Depends(get_db)) -> PredictionResponse:
    demo_user = get_or_create_user(db, telegram_user_id=0, username="api-demo")
    details = predict_match_details(features)
    decision = make_value_decision(
        probability=details.probability,
        odds=features.odds,
        bankroll=demo_user.bankroll,
        min_edge=settings.min_edge,
        max_kelly_fraction=settings.max_kelly_fraction,
    )
    return PredictionResponse(
        team=features.team,
        opponent=features.opponent,
        win_probability=round(details.probability, 4),
        implied_probability=round(decision.implied_probability, 4),
        value_edge=round(decision.value_edge, 4),
        kelly_fraction=round(decision.kelly_fraction, 4),
        suggested_stake=decision.suggested_stake,
        action=decision.action,
        exit_take_profit_odds=decision.exit_take_profit_odds,
        exit_stop_loss_odds=decision.exit_stop_loss_odds,
        explanation=decision.explanation,
        model_name=details.model_name,
        data_source=details.data_source,
        top_factors=details.top_factors,
        feature_values={key: round(value, 4) for key, value in details.feature_values.items()},
    )


@router.post("/bets/enter", response_model=BetResponse)
def enter(payload: EnterBetRequest, db: Session = Depends(get_db)) -> BetResponse:
    record = enter_bet(db, payload)
    return BetResponse.model_validate(record)


@router.get("/roi/{telegram_user_id}", response_model=RoiResponse)
def roi(telegram_user_id: int, db: Session = Depends(get_db)) -> RoiResponse:
    return get_roi_summary(db, telegram_user_id)


@router.post("/watchlist", response_model=WatchResponse)
def add_watch(payload: WatchRequest, db: Session = Depends(get_db)) -> WatchResponse:
    get_or_create_user(db, payload.telegram_user_id, payload.username)
    return WatchResponse.model_validate(add_watchlist_subscription(db, payload))


@router.get("/watchlist/{telegram_user_id}", response_model=list[WatchResponse])
def watchlist(telegram_user_id: int, db: Session = Depends(get_db)) -> list[WatchResponse]:
    return [
        WatchResponse.model_validate(item)
        for item in list_watchlist_subscriptions(db, telegram_user_id)
    ]


@router.delete("/watchlist/{telegram_user_id}/{match_id}/{team}")
def unwatch(telegram_user_id: int, match_id: str, team: str, db: Session = Depends(get_db)) -> dict:
    removed = disable_watchlist_subscription(db, telegram_user_id, match_id, team)
    return {"removed": removed}


@router.post("/scan/live", response_model=SignalResponse)
def scan_live(payload: LiveScanRequest, db: Session = Depends(get_db)) -> SignalResponse:
    get_or_create_user(db, payload.telegram_user_id, payload.username)
    subscription = add_watchlist_subscription(
        db,
        WatchRequest(
            telegram_user_id=payload.telegram_user_id,
            username=payload.username,
            match_id=payload.match_id,
            team=payload.team,
            opponent=payload.opponent,
            auto_trade=payload.auto_trade,
            notes=payload.notes,
        ),
    )
    live_state = live_data_provider.match_state(payload.match_id)
    if live_state is None:
        raise HTTPException(status_code=404, detail=f"No live state found for match {payload.match_id}")
    odds_snapshot = betfair_odds_provider.find_match_odds(payload.team, payload.opponent)
    if odds_snapshot is None:
        raise HTTPException(
            status_code=503,
            detail="No live odds were returned by the configured provider.",
        )
    signal, _ = run_watchlist_scan(db, subscription, live_state, odds_snapshot, notify=False)
    return SignalResponse.model_validate(signal)


@router.get("/signals/{telegram_user_id}", response_model=list[SignalResponse])
def signals(telegram_user_id: int, db: Session = Depends(get_db)) -> list[SignalResponse]:
    return [SignalResponse.model_validate(item) for item in list_recent_signals(db, telegram_user_id)]


@router.get("/demo/wallet/{telegram_user_id}", response_model=DemoWalletResponse)
def demo_wallet(telegram_user_id: int, db: Session = Depends(get_db)) -> DemoWalletResponse:
    return get_demo_wallet_summary(db, telegram_user_id)


@router.get("/demo/trades/{telegram_user_id}", response_model=list[DemoTradeResponse])
def demo_trades(telegram_user_id: int, db: Session = Depends(get_db)) -> list[DemoTradeResponse]:
    return [DemoTradeResponse.model_validate(item) for item in list_demo_trades(db, telegram_user_id)]
