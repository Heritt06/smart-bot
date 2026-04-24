from apscheduler.schedulers.background import BackgroundScheduler

from app.config import get_settings
from app.db import SessionLocal
from app.models import WatchlistSubscription
from app.services.demo_trading import run_watchlist_scan
from app.services.live_data import live_data_provider
from app.services.modeling import train_model_bundle
from app.services.odds_data import betfair_odds_provider
from sqlalchemy import select


settings = get_settings()
scheduler = BackgroundScheduler()


def refresh_match_data():
    if live_data_provider.configured():
        live_data_provider.current_matches()


def scan_watchlists():
    if not live_data_provider.configured() or not betfair_odds_provider.configured():
        return

    with SessionLocal() as db:
        subscriptions = db.scalars(
            select(WatchlistSubscription).where(WatchlistSubscription.enabled.is_(True))
        ).all()
        for subscription in subscriptions:
            live_state = live_data_provider.match_state(subscription.match_id)
            if live_state is None:
                continue
            odds_snapshot = betfair_odds_provider.find_match_odds(subscription.team, subscription.opponent)
            if odds_snapshot is None:
                continue
            run_watchlist_scan(db, subscription, live_state, odds_snapshot, notify=True)


def retrain_model():
    if settings.model_artifact.exists() or settings.ipl_data_zip.exists():
        train_model_bundle(refresh_data=False)


def start_scheduler() -> None:
    if not settings.scheduler_enabled or scheduler.running:
        return

    scheduler.add_job(refresh_match_data, "interval", minutes=15, id="refresh-match-data")
    scheduler.add_job(
        scan_watchlists,
        "interval",
        seconds=settings.live_scan_interval_seconds,
        id="scan-watchlists",
    )
    scheduler.add_job(retrain_model, "interval", hours=24, id="retrain-model")
    scheduler.start()


def stop_scheduler() -> None:
    if scheduler.running:
        scheduler.shutdown(wait=False)
