from __future__ import annotations

from datetime import datetime, timedelta, timezone

import requests

from app.constants import canonical_team_name
from app.config import get_settings
from app.schemas import OddsSnapshot


class OddsProviderError(RuntimeError):
    pass


class BetfairOddsProvider:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._session_token: str | None = None

    def configured(self) -> bool:
        required = (
            self.settings.betfair_app_key,
            self.settings.betfair_username,
            self.settings.betfair_password,
        )
        return all(required)

    def _ensure_login(self) -> str:
        if self._session_token:
            return self._session_token

        response = requests.post(
            self.settings.betfair_login_url,
            headers={
                "Accept": "application/json",
                "X-Application": self.settings.betfair_app_key,
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={
                "username": self.settings.betfair_username,
                "password": self.settings.betfair_password,
            },
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") not in {"SUCCESS", "LIMITED_ACCESS"} or not payload.get("token"):
            raise OddsProviderError(f"Betfair login failed: {payload}")
        self._session_token = payload["token"]
        return self._session_token

    def _rpc(self, method: str, params: dict) -> list[dict]:
        token = self._ensure_login()
        response = requests.post(
            self.settings.betfair_betting_url,
            headers={
                "X-Application": self.settings.betfair_app_key,
                "X-Authentication": token,
                "Content-Type": "application/json",
            },
            json=[
                {
                    "jsonrpc": "2.0",
                    "method": f"SportsAPING/v1.0/{method}",
                    "params": params,
                    "id": 1,
                }
            ],
            timeout=20,
        )
        response.raise_for_status()
        body = response.json()[0]
        if "error" in body:
            raise OddsProviderError(f"Betfair RPC failed for {method}: {body['error']}")
        return body.get("result") or []

    def find_match_odds(self, team: str, opponent: str) -> OddsSnapshot | None:
        if not self.configured():
            return None

        team = canonical_team_name(team)
        opponent = canonical_team_name(opponent)
        now = datetime.now(timezone.utc)
        filter_params = {
            "textQuery": team,
            "marketTypeCodes": ["MATCH_ODDS"],
            "turnInPlayEnabled": True,
            "marketStartTime": {
                "from": (now - timedelta(days=1)).isoformat(),
                "to": (now + timedelta(days=1)).isoformat(),
            },
        }
        catalogues = self._rpc(
            "listMarketCatalogue",
            {
                "filter": filter_params,
                "marketProjection": ["RUNNER_DESCRIPTION", "EVENT", "MARKET_START_TIME"],
                "sort": "FIRST_TO_START",
                "maxResults": "20",
                "locale": "en",
            },
        )

        selected_market = None
        selected_runner = None
        for market in catalogues:
            runners = market.get("runners") or []
            runner_names = {canonical_team_name(item.get("runnerName")): item for item in runners}
            if team in runner_names and opponent in runner_names:
                selected_market = market
                selected_runner = runner_names[team]
                break

        if selected_market is None or selected_runner is None:
            return None

        market_books = self._rpc(
            "listMarketBook",
            {
                "marketIds": [selected_market["marketId"]],
                "priceProjection": {
                    "priceData": ["EX_BEST_OFFERS", "EX_LTP"],
                    "virtualise": True,
                },
            },
        )
        if not market_books:
            return None

        runner_books = market_books[0].get("runners") or []
        selection_id = selected_runner.get("selectionId")
        runner_book = next(
            (item for item in runner_books if item.get("selectionId") == selection_id),
            None,
        )
        if runner_book is None:
            return None

        exchange = runner_book.get("ex") or {}
        available_to_back = exchange.get("availableToBack") or []
        available_to_lay = exchange.get("availableToLay") or []
        return OddsSnapshot(
            provider="betfair_exchange",
            market_id=selected_market["marketId"],
            selection_name=team,
            best_back_odds=available_to_back[0]["price"] if available_to_back else None,
            best_lay_odds=available_to_lay[0]["price"] if available_to_lay else None,
            last_traded_price=runner_book.get("lastPriceTraded"),
            total_matched=runner_book.get("totalMatched"),
            in_play=bool(market_books[0].get("inplay")),
            raw={"market": selected_market, "book": market_books[0], "runner": runner_book},
        )


betfair_odds_provider = BetfairOddsProvider()
