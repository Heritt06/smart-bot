from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class UserAccount(Base):
    __tablename__ = "user_accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    telegram_user_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    username: Mapped[str | None] = mapped_column(String(64), nullable=True)
    bankroll: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )


class BetRecord(Base):
    __tablename__ = "bet_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    telegram_user_id: Mapped[int] = mapped_column(Integer, index=True)
    username: Mapped[str | None] = mapped_column(String(64), nullable=True)
    market: Mapped[str] = mapped_column(String(64), default="match_winner")
    team: Mapped[str] = mapped_column(String(64))
    opponent: Mapped[str] = mapped_column(String(64))
    probability: Mapped[float] = mapped_column(Float)
    odds: Mapped[float] = mapped_column(Float)
    value_edge: Mapped[float] = mapped_column(Float)
    stake: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(32), default="open")
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class WatchlistSubscription(Base):
    __tablename__ = "watchlist_subscriptions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    telegram_user_id: Mapped[int] = mapped_column(Integer, index=True)
    username: Mapped[str | None] = mapped_column(String(64), nullable=True)
    match_id: Mapped[str] = mapped_column(String(64), index=True)
    team: Mapped[str] = mapped_column(String(64))
    opponent: Mapped[str] = mapped_column(String(64))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    auto_trade: Mapped[bool] = mapped_column(Boolean, default=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )


class SignalRecord(Base):
    __tablename__ = "signal_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    telegram_user_id: Mapped[int] = mapped_column(Integer, index=True)
    match_id: Mapped[str] = mapped_column(String(64), index=True)
    team: Mapped[str] = mapped_column(String(64))
    opponent: Mapped[str] = mapped_column(String(64))
    signal_type: Mapped[str] = mapped_column(String(32))
    odds: Mapped[float] = mapped_column(Float)
    probability: Mapped[float] = mapped_column(Float)
    value_edge: Mapped[float] = mapped_column(Float)
    message: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class DemoTrade(Base):
    __tablename__ = "demo_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    telegram_user_id: Mapped[int] = mapped_column(Integer, index=True)
    username: Mapped[str | None] = mapped_column(String(64), nullable=True)
    match_id: Mapped[str] = mapped_column(String(64), index=True)
    market_source: Mapped[str] = mapped_column(String(64), default="manual")
    team: Mapped[str] = mapped_column(String(64))
    opponent: Mapped[str] = mapped_column(String(64))
    entry_odds: Mapped[float] = mapped_column(Float)
    current_odds: Mapped[float] = mapped_column(Float)
    exit_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    stake: Mapped[float] = mapped_column(Float)
    probability_at_entry: Mapped[float] = mapped_column(Float)
    value_edge_at_entry: Mapped[float] = mapped_column(Float)
    exit_take_profit_odds: Mapped[float] = mapped_column(Float)
    exit_stop_loss_odds: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(32), default="open")
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    signal_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    opened_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
