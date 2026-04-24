from contextlib import suppress

from pydantic import ValidationError
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

from app.config import get_settings
from app.db import SessionLocal
from app.schemas import EnterBetRequest, MatchFeatures, PredictionResponse, WatchRequest
from app.services.live_data import live_data_provider
from app.services.modeling import load_model_bundle, train_model_bundle
from app.services.betting import enter_bet, get_or_create_user, get_roi_summary
from app.services.demo_trading import (
    add_watchlist_subscription,
    disable_watchlist_subscription,
    get_demo_wallet_summary,
    list_demo_trades,
    list_recent_signals,
    list_watchlist_subscriptions,
    run_watchlist_scan,
)
from app.services.odds_data import betfair_odds_provider
from app.services.prediction import predict_match_details
from app.services.value_engine import make_value_decision
from app.utils import as_bool, as_float, as_list, parse_key_value_pairs


settings = get_settings()


def build_prediction(features: MatchFeatures, bankroll: float) -> PredictionResponse:
    details = predict_match_details(features)
    decision = make_value_decision(
        probability=details.probability,
        odds=features.odds,
        bankroll=bankroll,
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


def format_prediction(prediction: PredictionResponse) -> str:
    return (
        f"{prediction.team} vs {prediction.opponent}\n"
        f"Win probability: {prediction.win_probability:.2%}\n"
        f"Implied probability: {prediction.implied_probability:.2%}\n"
        f"Value edge: {prediction.value_edge:.2%}\n"
        f"Kelly fraction: {prediction.kelly_fraction:.2%}\n"
        f"Suggested stake: {prediction.suggested_stake:.2f}\n"
        f"Action: {prediction.action.upper()}\n"
        f"Model: {prediction.model_name}\n"
        f"Take-profit odds: {prediction.exit_take_profit_odds}\n"
        f"Stop-loss odds: {prediction.exit_stop_loss_odds}\n"
        f"Why: {prediction.explanation}\n"
        f"Signals: {' | '.join(prediction.top_factors[:3])}"
    )


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user is None or update.message is None:
        return

    with SessionLocal() as db:
        user = get_or_create_user(
            db,
            telegram_user_id=update.effective_user.id,
            username=update.effective_user.username,
        )

    await update.message.reply_text(
        "IPL Smart Bets Bot is ready.\n"
        f"Bankroll initialized at {user.bankroll:.2f}.\n"
        "Use /help to see command examples."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    await update.message.reply_text(
        "Commands use key=value pairs.\n\n"
        "/predict team=MI opponent=CSK odds=2.10 venue=Chennai "
        "toss_winner=CSK toss_decision=field "
        "team_players=Rohit;Surya;Tilak;Hardik;TimDavid "
        "opponent_players=Gaikwad;Conway;Dube;Jadeja;Pathirana\n\n"
        "/enter team=MI opponent=CSK odds=2.10 stake=450 probability=0.58 market=match_winner\n\n"
        "/watch match_id=123456 team=MI opponent=CSK auto_trade=true\n\n"
        "/scan match_id=123456 team=MI opponent=CSK\n\n"
        "/watchlist\n\n"
        "/unwatch match_id=123456 team=MI\n\n"
        "/signals\n\n"
        "/wallet\n\n"
        "/trades\n\n"
        "/roi\n\n"
        "/model\n\n"
        "/train\n\n"
        "/matches"
    )


async def predict_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user is None or update.message is None:
        return

    try:
        values = parse_key_value_pairs(" ".join(context.args))
        features = MatchFeatures(
            team=values["team"],
            opponent=values["opponent"],
            odds=as_float(values, "odds"),
            venue=values.get("venue"),
            city=values.get("city"),
            match_date=values.get("match_date"),
            toss_winner=values.get("toss_winner"),
            toss_decision=values.get("toss_decision"),
            team_players=as_list(values, "team_players"),
            opponent_players=as_list(values, "opponent_players"),
            notes=values.get("notes"),
        )
    except (KeyError, ValueError, ValidationError) as exc:
        await update.message.reply_text(
            f"Could not parse /predict command: {exc}\nUse /help for the expected format."
        )
        return

    with SessionLocal() as db:
        user = get_or_create_user(
            db,
            telegram_user_id=update.effective_user.id,
            username=update.effective_user.username,
        )
        prediction = build_prediction(features, bankroll=user.bankroll)

    await update.message.reply_text(format_prediction(prediction))


async def enter_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user is None or update.message is None:
        return

    try:
        values = parse_key_value_pairs(" ".join(context.args))
        payload = EnterBetRequest(
            telegram_user_id=update.effective_user.id,
            username=update.effective_user.username,
            market=values.get("market", "match_winner"),
            team=values["team"],
            opponent=values["opponent"],
            probability=as_float(values, "probability"),
            odds=as_float(values, "odds"),
            stake=as_float(values, "stake"),
            notes=values.get("notes"),
        )
    except (KeyError, ValueError) as exc:
        await update.message.reply_text(
            f"Could not parse /enter command: {exc}\nUse /help for the expected format."
        )
        return

    with SessionLocal() as db:
        record = enter_bet(db, payload)

    await update.message.reply_text(
        f"Entry recorded.\nBet #{record.id}: {record.team} vs {record.opponent}\n"
        f"Stake: {record.stake:.2f}\nValue edge: {record.value_edge:.2%}"
    )


async def roi_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user is None or update.message is None:
        return

    with SessionLocal() as db:
        summary = get_roi_summary(db, update.effective_user.id)

    await update.message.reply_text(
        f"Bankroll: {summary.bankroll:.2f}\n"
        f"Invested: {summary.total_invested:.2f}\n"
        f"Profit: {summary.total_profit:.2f}\n"
        f"Open bets: {summary.open_bets}\n"
        f"Settled bets: {summary.settled_bets}\n"
        f"ROI: {summary.roi:.2%}"
    )


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    bundle = load_model_bundle()
    if bundle is None:
        await update.message.reply_text(
            "No trained model artifact found yet. Run /train to download IPL history and build one."
        )
        return

    metrics = bundle.metrics[bundle.model_name]
    await update.message.reply_text(
        f"Model: {bundle.model_name}\n"
        f"Trained at: {bundle.trained_at}\n"
        f"Samples: {bundle.sample_count}\n"
        f"Log loss: {metrics['log_loss']:.4f}\n"
        f"Accuracy: {metrics['accuracy']:.4f}\n"
        f"ROC AUC: {metrics['roc_auc']:.4f}"
    )


async def train_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    await update.message.reply_text("Training started. This may take a minute on the first run.")
    bundle = train_model_bundle(refresh_data=True)
    metrics = bundle.metrics[bundle.model_name]
    await update.message.reply_text(
        f"Training complete.\n"
        f"Best model: {bundle.model_name}\n"
        f"Samples: {bundle.sample_count}\n"
        f"Log loss: {metrics['log_loss']:.4f}\n"
        f"Accuracy: {metrics['accuracy']:.4f}\n"
        f"ROC AUC: {metrics['roc_auc']:.4f}"
    )


async def matches_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    matches = live_data_provider.current_matches()
    if not matches:
        await update.message.reply_text(
            "No live IPL matches were returned. Set CRICKET_DATA_API_KEY to enable live match listing."
        )
        return

    lines = ["Live IPL matches:"]
    for match in matches[:8]:
        lines.append(f"{match.match_id}: {match.title} | {match.status or 'status pending'}")
    await update.message.reply_text("\n".join(lines))


async def watch_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user is None or update.message is None:
        return
    try:
        values = parse_key_value_pairs(" ".join(context.args))
        payload = WatchRequest(
            telegram_user_id=update.effective_user.id,
            username=update.effective_user.username,
            match_id=values["match_id"],
            team=values["team"],
            opponent=values["opponent"],
            auto_trade=as_bool(values, "auto_trade", True),
            notes=values.get("notes"),
        )
    except (KeyError, ValueError, ValidationError) as exc:
        await update.message.reply_text(
            f"Could not parse /watch command: {exc}\nUse /help for the expected format."
        )
        return

    with SessionLocal() as db:
        get_or_create_user(db, payload.telegram_user_id, payload.username)
        record = add_watchlist_subscription(db, payload)

    await update.message.reply_text(
        f"Watching {record.team} vs {record.opponent} on match {record.match_id}.\n"
        f"Auto trade: {'ON' if record.auto_trade else 'OFF'}"
    )


async def watchlist_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user is None or update.message is None:
        return
    with SessionLocal() as db:
        records = list_watchlist_subscriptions(db, update.effective_user.id)
    if not records:
        await update.message.reply_text("Your watchlist is empty. Use /watch after /matches.")
        return
    lines = ["Active watchlist:"]
    for item in records[:10]:
        lines.append(
            f"{item.match_id}: {item.team} vs {item.opponent} | auto={'ON' if item.auto_trade else 'OFF'}"
        )
    await update.message.reply_text("\n".join(lines))


async def unwatch_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user is None or update.message is None:
        return
    try:
        values = parse_key_value_pairs(" ".join(context.args))
        match_id = values["match_id"]
        team = values["team"]
    except KeyError as exc:
        await update.message.reply_text(
            f"Could not parse /unwatch command: {exc}\nUse /help for the expected format."
        )
        return

    with SessionLocal() as db:
        removed = disable_watchlist_subscription(db, update.effective_user.id, match_id, team)
    await update.message.reply_text("Removed from watchlist." if removed else "Nothing matched that watch entry.")


async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user is None or update.message is None:
        return
    try:
        values = parse_key_value_pairs(" ".join(context.args))
        payload = WatchRequest(
            telegram_user_id=update.effective_user.id,
            username=update.effective_user.username,
            match_id=values["match_id"],
            team=values["team"],
            opponent=values["opponent"],
            auto_trade=as_bool(values, "auto_trade", True),
            notes=values.get("notes"),
        )
    except (KeyError, ValueError, ValidationError) as exc:
        await update.message.reply_text(
            f"Could not parse /scan command: {exc}\nUse /help for the expected format."
        )
        return

    live_state = live_data_provider.match_state(payload.match_id)
    if live_state is None:
        await update.message.reply_text("No live match state was returned for that match id.")
        return
    odds_snapshot = betfair_odds_provider.find_match_odds(payload.team, payload.opponent)
    if odds_snapshot is None:
        await update.message.reply_text(
            "No live odds were returned. Configure Betfair credentials to enable odds-based demo trading."
        )
        return

    with SessionLocal() as db:
        get_or_create_user(db, payload.telegram_user_id, payload.username)
        record = add_watchlist_subscription(db, payload)
        signal, trade = run_watchlist_scan(db, record, live_state, odds_snapshot, notify=False)

    message = signal.message
    if trade is not None and trade.status == "open":
        message = f"{message}\nDemo trade #{trade.id} is OPEN."
    elif trade is not None and trade.status == "closed":
        message = f"{message}\nDemo trade #{trade.id} is CLOSED."
    await update.message.reply_text(message)


async def signals_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user is None or update.message is None:
        return
    with SessionLocal() as db:
        signals = list_recent_signals(db, update.effective_user.id, limit=6)
    if not signals:
        await update.message.reply_text("No signals yet. Use /scan or /watch first.")
        return
    lines = ["Recent signals:"]
    for item in signals:
        lines.append(
            f"{item.signal_type.upper()} | {item.team} vs {item.opponent} | "
            f"odds {item.odds:.2f} | edge {item.value_edge:.2%}"
        )
    await update.message.reply_text("\n".join(lines))


async def wallet_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user is None or update.message is None:
        return
    with SessionLocal() as db:
        wallet = get_demo_wallet_summary(db, update.effective_user.id)
    await update.message.reply_text(
        f"Starting bankroll: {wallet.starting_bankroll:.2f}\n"
        f"Realized profit: {wallet.realized_profit:.2f}\n"
        f"Open exposure: {wallet.open_exposure:.2f}\n"
        f"Available balance: {wallet.available_balance:.2f}\n"
        f"Open trades: {wallet.open_trades}\n"
        f"Closed trades: {wallet.closed_trades}\n"
        f"ROI: {wallet.roi:.2%}"
    )


async def trades_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user is None or update.message is None:
        return
    with SessionLocal() as db:
        trades = list_demo_trades(db, update.effective_user.id)[:8]
    if not trades:
        await update.message.reply_text("No demo trades yet. Use /scan or /watch first.")
        return
    lines = ["Demo trades:"]
    for item in trades:
        pnl_text = f"{item.pnl:.2f}" if item.pnl is not None else "-"
        lines.append(
            f"#{item.id} {item.team} vs {item.opponent} | {item.status.upper()} | "
            f"entry {item.entry_odds:.2f} | current {item.current_odds:.2f} | pnl {pnl_text}"
        )
    await update.message.reply_text("\n".join(lines))


def create_telegram_application() -> Application | None:
    if not settings.telegram_bot_token:
        return None

    application = Application.builder().token(settings.telegram_bot_token).build()
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("predict", predict_command))
    application.add_handler(CommandHandler("enter", enter_command))
    application.add_handler(CommandHandler("roi", roi_command))
    application.add_handler(CommandHandler("model", model_command))
    application.add_handler(CommandHandler("train", train_command))
    application.add_handler(CommandHandler("matches", matches_command))
    application.add_handler(CommandHandler("watch", watch_command))
    application.add_handler(CommandHandler("watchlist", watchlist_command))
    application.add_handler(CommandHandler("unwatch", unwatch_command))
    application.add_handler(CommandHandler("scan", scan_command))
    application.add_handler(CommandHandler("signals", signals_command))
    application.add_handler(CommandHandler("wallet", wallet_command))
    application.add_handler(CommandHandler("trades", trades_command))
    return application


async def shutdown_telegram_application(application: Application | None) -> None:
    if application is None:
        return
    with suppress(RuntimeError):
        await application.stop()
    with suppress(RuntimeError):
        await application.shutdown()
