from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MatchFeatures(BaseModel):
    team: str = Field(..., description="Primary team or selection")
    opponent: str = Field(..., description="Opposing team")
    odds: float = Field(..., gt=1.0, description="Market decimal odds")
    venue: str | None = None
    city: str | None = None
    match_date: date | None = None
    toss_winner: str | None = None
    toss_decision: Literal["bat", "field"] | None = None
    team_players: list[str] = Field(default_factory=list)
    opponent_players: list[str] = Field(default_factory=list)
    notes: str | None = None


class PredictionResponse(BaseModel):
    team: str
    opponent: str
    win_probability: float
    implied_probability: float
    value_edge: float
    kelly_fraction: float
    suggested_stake: float
    action: str
    exit_take_profit_odds: float
    exit_stop_loss_odds: float
    explanation: str
    model_name: str
    data_source: str
    top_factors: list[str]
    feature_values: dict[str, float]


class EnterBetRequest(BaseModel):
    telegram_user_id: int
    username: str | None = None
    market: str = "match_winner"
    team: str
    opponent: str
    probability: float = Field(..., ge=0.0, le=1.0)
    odds: float = Field(..., gt=1.0)
    stake: float = Field(..., gt=0.0)
    notes: str | None = None


class BetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    team: str
    opponent: str
    status: str
    value_edge: float
    stake: float


class RoiResponse(BaseModel):
    telegram_user_id: int
    bankroll: float
    total_invested: float
    total_profit: float
    open_bets: int
    settled_bets: int
    roi: float


class LiveMatchResponse(BaseModel):
    match_id: str
    title: str
    status: str | None = None
    venue: str | None = None
    date: str | None = None


class LiveScoreEntry(BaseModel):
    inning: str | None = None
    runs: int | None = None
    wickets: int | None = None
    overs: float | None = None


class LiveMatchState(BaseModel):
    match_id: str
    title: str
    status: str | None = None
    venue: str | None = None
    date: str | None = None
    team_names: list[str] = Field(default_factory=list)
    score_entries: list[LiveScoreEntry] = Field(default_factory=list)
    batting_team: str | None = None
    bowling_team: str | None = None
    current_runs: int | None = None
    current_wickets: int | None = None
    current_overs: float | None = None
    target_runs: int | None = None
    runs_needed: int | None = None
    balls_remaining: int | None = None
    required_run_rate: float | None = None
    current_run_rate: float | None = None
    team_players: list[str] = Field(default_factory=list)
    opponent_players: list[str] = Field(default_factory=list)
    raw: dict = Field(default_factory=dict)


class OddsSnapshot(BaseModel):
    provider: str
    market_id: str
    selection_name: str
    best_back_odds: float | None = None
    best_lay_odds: float | None = None
    last_traded_price: float | None = None
    total_matched: float | None = None
    in_play: bool = False
    raw: dict = Field(default_factory=dict)


class WatchRequest(BaseModel):
    telegram_user_id: int
    username: str | None = None
    match_id: str
    team: str
    opponent: str
    auto_trade: bool = True
    notes: str | None = None


class LiveScanRequest(BaseModel):
    telegram_user_id: int
    username: str | None = None
    match_id: str
    team: str
    opponent: str
    auto_trade: bool = True
    notes: str | None = None


class WatchResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    match_id: str
    team: str
    opponent: str
    enabled: bool
    auto_trade: bool


class SignalResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    match_id: str
    team: str
    opponent: str
    signal_type: str
    odds: float
    probability: float
    value_edge: float
    message: str


class DemoTradeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    match_id: str
    team: str
    opponent: str
    status: str
    entry_odds: float
    current_odds: float
    exit_odds: float | None = None
    stake: float
    pnl: float | None = None


class DemoWalletResponse(BaseModel):
    telegram_user_id: int
    starting_bankroll: float
    realized_profit: float
    open_exposure: float
    available_balance: float
    open_trades: int
    closed_trades: int
    roi: float
