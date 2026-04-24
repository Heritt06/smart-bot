from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import BetRecord, UserAccount
from app.schemas import EnterBetRequest, RoiResponse
from app.services.value_engine import calculate_value


settings = get_settings()


def get_or_create_user(
    db: Session,
    telegram_user_id: int,
    username: str | None = None,
) -> UserAccount:
    user = db.scalar(
        select(UserAccount).where(UserAccount.telegram_user_id == telegram_user_id)
    )
    if user is None:
        user = UserAccount(
            telegram_user_id=telegram_user_id,
            username=username,
            bankroll=settings.default_bankroll,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    if username and user.username != username:
        user.username = username
        db.commit()
        db.refresh(user)
    return user


def enter_bet(db: Session, payload: EnterBetRequest) -> BetRecord:
    get_or_create_user(db, payload.telegram_user_id, payload.username)
    record = BetRecord(
        telegram_user_id=payload.telegram_user_id,
        username=payload.username,
        market=payload.market,
        team=payload.team,
        opponent=payload.opponent,
        probability=payload.probability,
        odds=payload.odds,
        value_edge=calculate_value(payload.probability, payload.odds),
        stake=payload.stake,
        notes=payload.notes,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_roi_summary(db: Session, telegram_user_id: int) -> RoiResponse:
    user = get_or_create_user(db, telegram_user_id)
    records = db.scalars(
        select(BetRecord).where(BetRecord.telegram_user_id == telegram_user_id)
    ).all()
    total_invested = sum(record.stake for record in records)
    total_profit = sum(record.pnl or 0.0 for record in records)
    open_bets = sum(1 for record in records if record.status == "open")
    settled_bets = sum(1 for record in records if record.status != "open")
    roi = (total_profit / total_invested) if total_invested else 0.0

    return RoiResponse(
        telegram_user_id=telegram_user_id,
        bankroll=user.bankroll,
        total_invested=round(total_invested, 2),
        total_profit=round(total_profit, 2),
        open_bets=open_bets,
        settled_bets=settled_bets,
        roi=round(roi, 4),
    )


def count_bets(db: Session) -> int:
    return db.scalar(select(func.count(BetRecord.id))) or 0
