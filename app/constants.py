TEAM_ALIASES = {
    "RCB": "Royal Challengers Bangalore",
    "ROYAL CHALLENGERS BENGALURU": "Royal Challengers Bangalore",
    "MI": "Mumbai Indians",
    "CSK": "Chennai Super Kings",
    "KKR": "Kolkata Knight Riders",
    "DC": "Delhi Capitals",
    "DD": "Delhi Capitals",
    "DELHI DAREDEVILS": "Delhi Capitals",
    "SRH": "Sunrisers Hyderabad",
    "RR": "Rajasthan Royals",
    "KXIP": "Punjab Kings",
    "PBKS": "Punjab Kings",
    "KINGS XI PUNJAB": "Punjab Kings",
    "GT": "Gujarat Titans",
    "LSG": "Lucknow Super Giants",
    "RISING PUNE SUPERGIANT": "Rising Pune Supergiants",
}

IPL_EVENT_NAMES = {
    "Indian Premier League",
}


def canonical_team_name(name: str | None) -> str:
    if not name:
        return ""
    cleaned = " ".join(name.strip().split())
    return TEAM_ALIASES.get(cleaned.upper(), cleaned)
