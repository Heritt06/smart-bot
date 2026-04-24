from __future__ import annotations

import re
from typing import Any

import requests

from app.config import get_settings
from app.schemas import LiveMatchResponse, LiveMatchState, LiveScoreEntry


def _extract_match_list(payload: dict) -> list[dict]:
    for key in ("data", "matches", "response"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return [value]
    return []


def _extract_team_names(item: dict) -> list[str]:
    team_info = item.get("teamInfo") or item.get("teams") or []
    if team_info and isinstance(team_info[0], dict):
        names = [team.get("name") or team.get("shortname") for team in team_info]
        return [name for name in names if name]
    return [str(team) for team in team_info if team]


def _parse_score_string(score_text: str) -> LiveScoreEntry:
    runs = wickets = None
    overs = None
    innings_name = score_text.strip()
    slash_match = re.search(r"(\d+)\s*/\s*(\d+)", score_text)
    overs_match = re.search(r"\((\d+(?:\.\d+)?)\)", score_text)
    if slash_match:
        runs = int(slash_match.group(1))
        wickets = int(slash_match.group(2))
    if overs_match:
        overs = float(overs_match.group(1))
    innings_name = re.sub(r"\s+\d+\s*/\s*\d+.*$", "", innings_name).strip(" -")
    return LiveScoreEntry(inning=innings_name or None, runs=runs, wickets=wickets, overs=overs)


def _parse_score_entries(item: dict) -> list[LiveScoreEntry]:
    entries: list[LiveScoreEntry] = []
    raw_scores = item.get("score") or item.get("scores") or []
    for raw in raw_scores:
        if isinstance(raw, str):
            entries.append(_parse_score_string(raw))
            continue
        entries.append(
            LiveScoreEntry(
                inning=raw.get("inning") or raw.get("inningName"),
                runs=raw.get("r") or raw.get("runs"),
                wickets=raw.get("w") or raw.get("wkts") or raw.get("wickets"),
                overs=raw.get("o") or raw.get("overs"),
            )
        )
    return entries


def _score_overs_to_balls(overs: float | None) -> int | None:
    if overs is None:
        return None
    completed_overs = int(overs)
    balls_part = int(round((overs - completed_overs) * 10))
    return (completed_overs * 6) + balls_part


def _build_live_state(item: dict) -> LiveMatchState:
    score_entries = _parse_score_entries(item)
    team_names = _extract_team_names(item)
    batting_team = None
    bowling_team = None
    current_runs = current_wickets = None
    current_overs = None
    target_runs = runs_needed = balls_remaining = None
    required_run_rate = current_run_rate = None

    if score_entries:
        latest = score_entries[-1]
        batting_team = latest.inning or (team_names[0] if team_names else None)
        current_runs = latest.runs
        current_wickets = latest.wickets
        current_overs = latest.overs
        current_balls = _score_overs_to_balls(current_overs)
        if current_runs is not None and current_balls and current_balls > 0:
            current_run_rate = (current_runs / current_balls) * 6.0

        if len(score_entries) >= 2 and score_entries[0].runs is not None and current_balls is not None:
            target_runs = score_entries[0].runs + 1
            runs_needed = max(0, target_runs - (current_runs or 0))
            balls_remaining = max(0, 120 - current_balls)
            if balls_remaining > 0:
                required_run_rate = runs_needed / balls_remaining * 6.0

    if team_names and batting_team:
        normalized = batting_team.lower()
        bowling_team = next((team for team in team_names if team.lower() not in normalized), None)

    raw = dict(item)
    return LiveMatchState(
        match_id=str(item.get("id") or item.get("unique_id") or item.get("match_id") or ""),
        title=str(item.get("name") or item.get("title") or "IPL Match"),
        status=item.get("status") or item.get("description"),
        venue=item.get("venue"),
        date=item.get("date") or item.get("dateTimeGMT"),
        team_names=team_names,
        score_entries=score_entries,
        batting_team=batting_team,
        bowling_team=bowling_team,
        current_runs=current_runs,
        current_wickets=current_wickets,
        current_overs=current_overs,
        target_runs=target_runs,
        runs_needed=runs_needed,
        balls_remaining=balls_remaining,
        required_run_rate=required_run_rate,
        current_run_rate=current_run_rate,
        team_players=[],
        opponent_players=[],
        raw=raw,
    )


class CricketDataProvider:
    def __init__(self) -> None:
        self.settings = get_settings()

    def configured(self) -> bool:
        return bool(self.settings.cricket_data_api_key)

    def _get(self, endpoint: str, **params: Any) -> dict:
        response = requests.get(
            f"{self.settings.cricket_data_base_url.rstrip('/')}/{endpoint}",
            params={"apikey": self.settings.cricket_data_api_key, **params},
            timeout=20,
        )
        response.raise_for_status()
        return response.json()

    def current_matches(self) -> list[LiveMatchResponse]:
        if not self.configured():
            return []

        payload = self._get("currentMatches", offset=0)
        matches: list[LiveMatchResponse] = []
        for item in _extract_match_list(payload):
            text_blob = " ".join(
                str(item.get(field, "")) for field in ("name", "series", "status", "matchType", "title")
            )
            if "ipl" not in text_blob.lower() and "indian premier league" not in text_blob.lower():
                continue
            match_id = str(item.get("id") or item.get("unique_id") or item.get("match_id") or "")
            if not match_id:
                continue
            matches.append(
                LiveMatchResponse(
                    match_id=match_id,
                    title=str(item.get("name") or item.get("title") or "IPL Match"),
                    status=item.get("status") or item.get("description"),
                    venue=item.get("venue"),
                    date=item.get("date") or item.get("dateTimeGMT"),
                )
            )
        return matches

    def match_state(self, match_id: str) -> LiveMatchState | None:
        if not self.configured():
            return None

        for endpoint in ("match_scorecard", "currentMatches", "cricScore"):
            try:
                payload = self._get(endpoint, id=match_id, offset=0)
            except requests.RequestException:
                continue
            items = _extract_match_list(payload)
            if endpoint != "match_scorecard":
                items = [item for item in items if str(item.get("id") or item.get("unique_id") or item.get("match_id")) == match_id]
            if not items:
                continue
            return _build_live_state(items[0])
        return None


live_data_provider = CricketDataProvider()
