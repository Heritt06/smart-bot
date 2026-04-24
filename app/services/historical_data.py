from __future__ import annotations

import json
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from statistics import mean
from urllib.request import urlopen

from app.config import get_settings
from app.constants import IPL_EVENT_NAMES, canonical_team_name
from app.schemas import MatchFeatures


FEATURE_ORDER = [
    "recent_win_rate_diff",
    "recent_runs_scored_diff",
    "recent_runs_conceded_diff",
    "recent_run_rate_diff",
    "recent_wickets_taken_diff",
    "recent_wickets_lost_diff",
    "head_to_head_win_rate",
    "venue_team_win_rate",
    "venue_opponent_win_rate",
    "venue_first_innings_avg",
    "venue_chase_win_rate",
    "rest_days_diff",
    "batting_strength_diff",
    "bowling_strength_diff",
    "lineup_experience_diff",
    "batter_bowler_matchup_diff",
    "toss_won_by_team",
    "toss_won_by_opponent",
    "toss_decision_bat",
    "toss_decision_field",
    "toss_advantage_team",
    "known_team_players",
    "known_opponent_players",
]


def _safe_mean(values: list[float], default: float) -> float:
    return mean(values) if values else default


def _clip_history(items: list, limit: int) -> None:
    if len(items) > limit:
        del items[:-limit]


def _get_match_date(info: dict) -> date:
    dates = info.get("dates") or []
    if not dates:
        return date.today()
    return datetime.fromisoformat(dates[0]).date()


def _flatten_deliveries(innings: dict) -> list[dict]:
    deliveries: list[dict] = []
    if "overs" in innings:
        for over in innings.get("overs", []):
            deliveries.extend(over.get("deliveries", []))
    else:
        deliveries.extend(innings.get("deliveries", []))
    return deliveries


def _bowler_gets_credit(kind: str) -> bool:
    return kind not in {"run out", "retired hurt", "retired out", "obstructing the field"}


def _default_innings_stats() -> dict:
    return {
        "runs": 0,
        "balls": 0,
        "wickets_lost": 0,
        "batting": defaultdict(lambda: {"runs": 0, "balls": 0, "outs": 0}),
        "bowling": defaultdict(
            lambda: {"balls": 0, "runs_conceded": 0, "wickets": 0, "dot_balls": 0}
        ),
        "matchups": defaultdict(lambda: {"balls": 0, "runs": 0, "dismissals": 0}),
    }


def download_ipl_dataset(force: bool = False) -> Path:
    settings = get_settings()
    target = settings.ipl_data_zip
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        return target

    with urlopen(settings.cricsheet_ipl_url, timeout=60) as response:
        target.write_bytes(response.read())
    return target


def load_ipl_matches(zip_path: Path | None = None) -> list[dict]:
    settings = get_settings()
    archive = zip_path or settings.ipl_data_zip
    if not archive.exists():
        raise FileNotFoundError(
            f"IPL data archive not found at {archive}. Run the training script with refresh enabled."
        )

    parsed_matches: list[dict] = []
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            if not name.endswith(".json"):
                continue

            raw_match = json.loads(zf.read(name))
            info = raw_match.get("info", {})
            event_name = (info.get("event") or {}).get("name")
            if event_name not in IPL_EVENT_NAMES:
                continue

            teams_raw = info.get("teams") or []
            if len(teams_raw) != 2:
                continue

            team_map = {raw: canonical_team_name(raw) for raw in teams_raw}
            teams = [team_map[teams_raw[0]], team_map[teams_raw[1]]]
            innings_stats = {team: _default_innings_stats() for team in teams}

            players = {}
            for raw_team, player_list in (info.get("players") or {}).items():
                players[canonical_team_name(raw_team)] = list(player_list)

            first_innings_team = ""
            for index, innings in enumerate(raw_match.get("innings", [])):
                batting_team = canonical_team_name(innings.get("team"))
                if batting_team not in innings_stats:
                    continue
                if index == 0:
                    first_innings_team = batting_team

                stats = innings_stats[batting_team]
                for delivery in _flatten_deliveries(innings):
                    batter = delivery.get("batter")
                    bowler = delivery.get("bowler")
                    runs = delivery.get("runs") or {}
                    extras = delivery.get("extras") or {}
                    batter_runs = int(runs.get("batter", 0))
                    total_runs = int(runs.get("total", 0))
                    legal_delivery = "wides" not in extras and "noballs" not in extras

                    stats["runs"] += total_runs
                    if legal_delivery:
                        stats["balls"] += 1
                    if batter:
                        stats["batting"][batter]["runs"] += batter_runs
                        if legal_delivery:
                            stats["batting"][batter]["balls"] += 1
                    if bowler:
                        stats["bowling"][bowler]["runs_conceded"] += total_runs
                        if legal_delivery:
                            stats["bowling"][bowler]["balls"] += 1
                            stats["matchups"][(batter, bowler)]["balls"] += 1
                            if total_runs == 0:
                                stats["bowling"][bowler]["dot_balls"] += 1
                        stats["matchups"][(batter, bowler)]["runs"] += batter_runs

                    for wicket in delivery.get("wickets", []):
                        player_out = wicket.get("player_out")
                        kind = wicket.get("kind", "")
                        stats["wickets_lost"] += 1
                        if player_out:
                            stats["batting"][player_out]["outs"] += 1
                        if bowler and batter and _bowler_gets_credit(kind):
                            stats["bowling"][bowler]["wickets"] += 1
                            stats["matchups"][(batter, bowler)]["dismissals"] += 1

            if first_innings_team:
                second_innings_team = next(team for team in teams if team != first_innings_team)
                innings_stats[first_innings_team]["runs_conceded"] = innings_stats[second_innings_team][
                    "runs"
                ]
                innings_stats[first_innings_team]["wickets_taken"] = innings_stats[
                    second_innings_team
                ]["wickets_lost"]
                innings_stats[second_innings_team]["runs_conceded"] = innings_stats[first_innings_team][
                    "runs"
                ]
                innings_stats[second_innings_team]["wickets_taken"] = innings_stats[
                    first_innings_team
                ]["wickets_lost"]

            outcome = info.get("outcome") or {}
            winner = canonical_team_name(outcome.get("winner")) if outcome.get("winner") else None
            parsed_matches.append(
                {
                    "date": _get_match_date(info),
                    "season": str(info.get("season", "")),
                    "team1": teams[0],
                    "team2": teams[1],
                    "winner": winner,
                    "venue": info.get("venue") or "Unknown Venue",
                    "city": info.get("city"),
                    "toss_winner": canonical_team_name((info.get("toss") or {}).get("winner")),
                    "toss_decision": (info.get("toss") or {}).get("decision"),
                    "players": players,
                    "team_stats": innings_stats,
                    "first_innings_team": first_innings_team,
                    "first_innings_runs": innings_stats.get(first_innings_team, {}).get("runs", 0),
                    "chased_successfully": bool(
                        winner and first_innings_team and winner != first_innings_team
                    ),
                }
            )

    parsed_matches.sort(key=lambda match: match["date"])
    return parsed_matches


@dataclass
class FeatureRepository:
    team_history: dict[str, list[dict]] = field(default_factory=lambda: defaultdict(list))
    venue_history: dict[str, list[dict]] = field(default_factory=lambda: defaultdict(list))
    h2h_history: dict[tuple[str, str], list[int]] = field(
        default_factory=lambda: defaultdict(list)
    )
    player_batting_history: dict[str, list[dict]] = field(
        default_factory=lambda: defaultdict(list)
    )
    player_bowling_history: dict[str, list[dict]] = field(
        default_factory=lambda: defaultdict(list)
    )
    batter_bowler_history: dict[tuple[str, str], dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: {"balls": 0, "runs": 0, "dismissals": 0})
    )
    team_lineups: dict[str, list[list[str]]] = field(default_factory=lambda: defaultdict(list))

    def to_state(self) -> dict:
        return {
            "team_history": dict(self.team_history),
            "venue_history": dict(self.venue_history),
            "h2h_history": dict(self.h2h_history),
            "player_batting_history": dict(self.player_batting_history),
            "player_bowling_history": dict(self.player_bowling_history),
            "batter_bowler_history": dict(self.batter_bowler_history),
            "team_lineups": dict(self.team_lineups),
        }

    @classmethod
    def from_state(cls, state: dict) -> "FeatureRepository":
        repository = cls()
        repository.team_history.update(state.get("team_history", {}))
        repository.venue_history.update(state.get("venue_history", {}))
        repository.h2h_history.update(state.get("h2h_history", {}))
        repository.player_batting_history.update(state.get("player_batting_history", {}))
        repository.player_bowling_history.update(state.get("player_bowling_history", {}))
        repository.batter_bowler_history.update(state.get("batter_bowler_history", {}))
        repository.team_lineups.update(state.get("team_lineups", {}))
        return repository

    def _recent_team_history(self, team: str, limit: int = 5) -> list[dict]:
        return self.team_history.get(team, [])[-limit:]

    def _infer_probable_lineup(self, team: str) -> list[str]:
        lineups = self.team_lineups.get(team, [])[-5:]
        if not lineups:
            return []
        counter = Counter(player for lineup in lineups for player in lineup)
        return [player for player, _ in counter.most_common(11)]

    def _player_batting_rating(self, player: str) -> float:
        history = self.player_batting_history.get(player, [])[-10:]
        if not history:
            return 20.0
        total_runs = sum(item["runs"] for item in history)
        total_balls = sum(item["balls"] for item in history)
        dismissals = sum(item["outs"] for item in history)
        average_runs = total_runs / len(history)
        strike_rate = 100.0 * total_runs / max(total_balls, 1)
        dismissal_penalty = dismissals / max(len(history), 1)
        return average_runs * 0.7 + strike_rate * 0.12 - dismissal_penalty * 6.0

    def _player_bowling_rating(self, player: str) -> float:
        history = self.player_bowling_history.get(player, [])[-10:]
        if not history:
            return 0.0
        wickets = sum(item["wickets"] for item in history)
        balls = sum(item["balls"] for item in history)
        runs_conceded = sum(item["runs_conceded"] for item in history)
        dot_balls = sum(item["dot_balls"] for item in history)
        wickets_per_innings = wickets / len(history)
        economy = 6.0 * runs_conceded / max(balls, 1)
        dot_rate = dot_balls / max(balls, 1)
        return wickets_per_innings * 18.0 + dot_rate * 14.0 - economy * 2.5

    def _team_metrics(self, team: str, venue: str | None, match_date: date) -> dict[str, float]:
        recent = self._recent_team_history(team)
        recent_runs_scored = _safe_mean([item["runs_scored"] for item in recent], 165.0)
        recent_runs_conceded = _safe_mean([item["runs_conceded"] for item in recent], 165.0)
        recent_run_rate = _safe_mean([item["run_rate"] for item in recent], 8.25)
        recent_wickets_taken = _safe_mean([item["wickets_taken"] for item in recent], 6.0)
        recent_wickets_lost = _safe_mean([item["wickets_lost"] for item in recent], 6.0)
        recent_win_rate = _safe_mean([item["won"] for item in recent], 0.5)
        last_match_date = recent[-1]["date"] if recent else match_date
        venue_history = [
            item for item in self.team_history.get(team, []) if venue and item["venue"] == venue
        ]
        venue_win_rate = _safe_mean([item["won"] for item in venue_history[-5:]], recent_win_rate)

        return {
            "win_rate": recent_win_rate,
            "avg_runs_scored": recent_runs_scored,
            "avg_runs_conceded": recent_runs_conceded,
            "run_rate": recent_run_rate,
            "avg_wickets_taken": recent_wickets_taken,
            "avg_wickets_lost": recent_wickets_lost,
            "venue_win_rate": venue_win_rate,
            "rest_days": float((match_date - last_match_date).days if recent else 7),
        }

    def _venue_metrics(self, venue: str | None) -> dict[str, float]:
        if not venue:
            return {"first_innings_avg": 170.0, "chase_win_rate": 0.5}
        history = self.venue_history.get(venue, [])
        return {
            "first_innings_avg": _safe_mean(
                [item["first_innings_runs"] for item in history[-20:]], 170.0
            ),
            "chase_win_rate": _safe_mean([item["chased_successfully"] for item in history[-20:]], 0.5),
        }

    def _head_to_head(self, team: str, opponent: str) -> float:
        return _safe_mean(self.h2h_history.get((team, opponent), []), 0.5)

    def _lineup_strength(self, lineup: list[str]) -> tuple[float, float, float]:
        if not lineup:
            return 20.0, 0.0, 0.0
        batting_scores = sorted(
            (self._player_batting_rating(player) for player in lineup),
            reverse=True,
        )
        bowling_scores = sorted(
            (self._player_bowling_rating(player) for player in lineup),
            reverse=True,
        )
        experience = _safe_mean(
            [
                float(
                    max(
                        len(self.player_batting_history.get(player, [])),
                        len(self.player_bowling_history.get(player, [])),
                    )
                )
                for player in lineup
            ],
            0.0,
        )
        return (
            _safe_mean(batting_scores[:7], 20.0),
            _safe_mean(bowling_scores[:5], 0.0),
            experience,
        )

    def _matchup_edge(self, batting_lineup: list[str], bowling_lineup: list[str]) -> float:
        if not batting_lineup or not bowling_lineup:
            return 0.0
        scores: list[float] = []
        for batter in batting_lineup[:7]:
            for bowler in bowling_lineup[:5]:
                matchup = self.batter_bowler_history.get((batter, bowler))
                if not matchup or matchup["balls"] < 3:
                    continue
                strike_rate = 100.0 * matchup["runs"] / max(matchup["balls"], 1)
                dismissal_rate = matchup["dismissals"] / max(matchup["balls"], 1)
                scores.append(strike_rate * 0.04 - dismissal_rate * 22.0)
        return _safe_mean(scores, 0.0)

    def build_feature_dict(self, request: MatchFeatures) -> dict[str, float]:
        team = canonical_team_name(request.team)
        opponent = canonical_team_name(request.opponent)
        venue = request.venue
        match_date = request.match_date or date.today()

        team_lineup = request.team_players or self._infer_probable_lineup(team)
        opponent_lineup = request.opponent_players or self._infer_probable_lineup(opponent)

        team_metrics = self._team_metrics(team, venue, match_date)
        opponent_metrics = self._team_metrics(opponent, venue, match_date)
        venue_metrics = self._venue_metrics(venue)
        team_batting, team_bowling, team_experience = self._lineup_strength(team_lineup)
        opponent_batting, opponent_bowling, opponent_experience = self._lineup_strength(
            opponent_lineup
        )
        matchup_for_team = self._matchup_edge(team_lineup, opponent_lineup)
        matchup_against_team = self._matchup_edge(opponent_lineup, team_lineup)

        toss_winner = canonical_team_name(request.toss_winner)
        toss_decision = request.toss_decision
        toss_advantage = 0.0
        if toss_winner:
            chase_edge = venue_metrics["chase_win_rate"]
            if toss_decision == "field":
                toss_advantage = chase_edge if toss_winner == team else -chase_edge
            elif toss_decision == "bat":
                bat_first_edge = 1.0 - chase_edge
                toss_advantage = bat_first_edge if toss_winner == team else -bat_first_edge

        feature_values = {
            "recent_win_rate_diff": team_metrics["win_rate"] - opponent_metrics["win_rate"],
            "recent_runs_scored_diff": team_metrics["avg_runs_scored"]
            - opponent_metrics["avg_runs_scored"],
            "recent_runs_conceded_diff": opponent_metrics["avg_runs_conceded"]
            - team_metrics["avg_runs_conceded"],
            "recent_run_rate_diff": team_metrics["run_rate"] - opponent_metrics["run_rate"],
            "recent_wickets_taken_diff": team_metrics["avg_wickets_taken"]
            - opponent_metrics["avg_wickets_taken"],
            "recent_wickets_lost_diff": opponent_metrics["avg_wickets_lost"]
            - team_metrics["avg_wickets_lost"],
            "head_to_head_win_rate": self._head_to_head(team, opponent),
            "venue_team_win_rate": team_metrics["venue_win_rate"],
            "venue_opponent_win_rate": opponent_metrics["venue_win_rate"],
            "venue_first_innings_avg": venue_metrics["first_innings_avg"],
            "venue_chase_win_rate": venue_metrics["chase_win_rate"],
            "rest_days_diff": team_metrics["rest_days"] - opponent_metrics["rest_days"],
            "batting_strength_diff": team_batting - opponent_batting,
            "bowling_strength_diff": team_bowling - opponent_bowling,
            "lineup_experience_diff": team_experience - opponent_experience,
            "batter_bowler_matchup_diff": matchup_for_team - matchup_against_team,
            "toss_won_by_team": float(toss_winner == team),
            "toss_won_by_opponent": float(toss_winner == opponent),
            "toss_decision_bat": float(toss_decision == "bat"),
            "toss_decision_field": float(toss_decision == "field"),
            "toss_advantage_team": toss_advantage,
            "known_team_players": float(len(team_lineup)),
            "known_opponent_players": float(len(opponent_lineup)),
        }
        return feature_values

    def update_from_match(self, match: dict) -> None:
        teams = [match["team1"], match["team2"]]
        winner = match["winner"]

        for team in teams:
            stats = match["team_stats"].get(team)
            if not stats:
                continue

            lineup = list(match["players"].get(team, []))
            team_entry = {
                "date": match["date"],
                "won": 1 if winner == team else 0,
                "runs_scored": stats["runs"],
                "runs_conceded": stats.get("runs_conceded", 0),
                "run_rate": 6.0 * stats["runs"] / max(stats["balls"], 1),
                "wickets_taken": stats.get("wickets_taken", 0),
                "wickets_lost": stats["wickets_lost"],
                "venue": match["venue"],
                "lineup": lineup,
            }
            self.team_history[team].append(team_entry)
            self.team_lineups[team].append(lineup)
            _clip_history(self.team_history[team], 100)
            _clip_history(self.team_lineups[team], 12)

            for player, batting in stats["batting"].items():
                self.player_batting_history[player].append(batting.copy())
                _clip_history(self.player_batting_history[player], 30)
            for player, bowling in stats["bowling"].items():
                self.player_bowling_history[player].append(bowling.copy())
                _clip_history(self.player_bowling_history[player], 30)
            for pair, matchup in stats["matchups"].items():
                target = self.batter_bowler_history[pair]
                target["balls"] += matchup["balls"]
                target["runs"] += matchup["runs"]
                target["dismissals"] += matchup["dismissals"]

        if winner in teams:
            self.h2h_history[(match["team1"], match["team2"])].append(
                1 if winner == match["team1"] else 0
            )
            self.h2h_history[(match["team2"], match["team1"])].append(
                1 if winner == match["team2"] else 0
            )
            _clip_history(self.h2h_history[(match["team1"], match["team2"])], 30)
            _clip_history(self.h2h_history[(match["team2"], match["team1"])], 30)

        if match["venue"]:
            self.venue_history[match["venue"]].append(
                {
                    "first_innings_runs": match["first_innings_runs"],
                    "chased_successfully": 1 if match["chased_successfully"] else 0,
                }
            )
            _clip_history(self.venue_history[match["venue"]], 40)


@dataclass
class TrainingSample:
    feature_values: dict[str, float]
    label: int
    match_date: date
    team: str
    opponent: str


def build_training_samples(matches: list[dict]) -> tuple[list[TrainingSample], FeatureRepository]:
    repository = FeatureRepository()
    samples: list[TrainingSample] = []

    for match in matches:
        if match["winner"] in {match["team1"], match["team2"]}:
            for team, opponent in ((match["team1"], match["team2"]), (match["team2"], match["team1"])):
                request = MatchFeatures(
                    team=team,
                    opponent=opponent,
                    odds=2.0,
                    venue=match["venue"],
                    city=match["city"],
                    match_date=match["date"],
                    toss_winner=match["toss_winner"] or None,
                    toss_decision=match["toss_decision"],
                    team_players=match["players"].get(team, []),
                    opponent_players=match["players"].get(opponent, []),
                )
                samples.append(
                    TrainingSample(
                        feature_values=repository.build_feature_dict(request),
                        label=1 if match["winner"] == team else 0,
                        match_date=match["date"],
                        team=team,
                        opponent=opponent,
                    )
                )

        repository.update_from_match(match)

    return samples, repository
