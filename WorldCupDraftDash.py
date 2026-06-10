import io
import logging
import re
import shutil
from pathlib import Path

import colorlover as cl
import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output, State

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_DIR = Path(__file__).parent / "log"
_LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WorldCupDraftDash")
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    _file_handler = logging.FileHandler(_LOG_DIR / "worldcup_dash.log", encoding="utf-8")
    _file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_file_handler)
    _stream_handler = logging.StreamHandler()
    _stream_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(_stream_handler)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_YEAR: int = 2026
AVAILABLE_YEARS: list = [2016, 2018, 2020, 2022, 2024, 2026]
ASSETS_DIR: Path = Path(__file__).parent / "assets"
CACHE_DIR: Path = Path(__file__).parent / "cache"

ISO_3166_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes"
FLAG_CDN_BASE_URL = "https://flagcdn.com/w80/"
PNG_HEADER_BYTES = b"\x89PNG\r\n\x1a\n"

GB_SUBDIVISION_CODES = {
    "GB-ENG": "England",
    "GB-SCT": "Scotland",
    "GB-WLS": "Wales",
    "GB-NIR": "Northern Ireland",
}

EVENT_MAP = {
    "assist": "assist",
    "scored penalty (shootout)": "penalty",
    "red card": "red",
    "penalty save (incl. shootout)": "save",
    "own goal": "og",
    "missed/saved penalty (incl. shootout)": "miss",
    "goal (excl. shootout)": "goal",
    "clean sheet (full time *)": "cs",
}

POSITION_FILTER_OPTIONS = [
    {"label": "All", "value": "all"},
    {"label": "GK", "value": "GK"},
    {"label": "DF", "value": "DF"},
    {"label": "MF", "value": "MF"},
    {"label": "ST/FW", "value": "ST"},
]

CHART_TABLE_TOGGLE_OPTIONS = [
    {"label": "Chart", "value": "chart"},
    {"label": "Table", "value": "table"},
]

PLAYER_STATS_SUMMARY_COLUMNS = [
    {"name": "Player", "id": "Player"},
    {"name": "Position", "id": "Position"},
    {"name": "Country", "id": "Country"},
    {"name": "Points", "id": "Points", "type": "numeric"},
    {"name": "Appearances", "id": "Appearances", "type": "numeric"},
    {"name": "Goals", "id": "Goals", "type": "numeric"},
    {"name": "Pen. Goals", "id": "Pen. Goals", "type": "numeric"},
    {"name": "Assists", "id": "Assists", "type": "numeric"},
    {"name": "Clean Sheets", "id": "Clean Sheets", "type": "numeric"},
    {"name": "Pen. Saves", "id": "Pen. Saves", "type": "numeric"},
    {"name": "Own Goals", "id": "Own Goals", "type": "numeric"},
    {"name": "Red Cards", "id": "Red Cards", "type": "numeric"},
    {"name": "Missed Pens.", "id": "Missed Pens.", "type": "numeric"},
]

PLAYER_STATS_GW_COLUMNS = [
    {"name": "Gameweek", "id": "Gameweek", "type": "numeric"},
    {"name": "Points", "id": "Points", "type": "numeric"},
    {"name": "Goals", "id": "Goals", "type": "numeric"},
    {"name": "Pen. Goals", "id": "Pen. Goals", "type": "numeric"},
    {"name": "Assists", "id": "Assists", "type": "numeric"},
    {"name": "Clean Sheets", "id": "Clean Sheets", "type": "numeric"},
    {"name": "Pen. Saves", "id": "Pen. Saves", "type": "numeric"},
    {"name": "Own Goals", "id": "Own Goals", "type": "numeric"},
    {"name": "Red Cards", "id": "Red Cards", "type": "numeric"},
    {"name": "Missed Pens.", "id": "Missed Pens.", "type": "numeric"},
]

PITCH_WIDTH = 100
PITCH_HEIGHT = 100

POSITION_LINE_MAP = {
    "GK":  (PITCH_HEIGHT / 10,       1),
    "GKP": (PITCH_HEIGHT / 10,       1),
    "DF":  (3.5 * PITCH_HEIGHT / 10, 4),
    "DEF": (3.5 * PITCH_HEIGHT / 10, 4),
    "MF":  (6.2 * PITCH_HEIGHT / 10, 4),
    "MID": (6.2 * PITCH_HEIGHT / 10, 4),
    "ST":  (8.7 * PITCH_HEIGHT / 10, 3),
    "FW":  (8.7 * PITCH_HEIGHT / 10, 3),
    "FWD": (8.7 * PITCH_HEIGHT / 10, 3),
}

COLOUR_DARK_GREY = "#363434"
COLOUR_WHITE = "white"
COLOUR_BLACK = "#121212"
COLOUR_TRANSPARENT = "rgba(0, 0, 0, 0)"

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_current_year: int = DEFAULT_YEAR
_merged_df: pd.DataFrame = pd.DataFrame()
_user_options: list = []
_gameweek_options: list = ["All"]
_transfer_dates: list = []

# ---------------------------------------------------------------------------
# URL builders
# ---------------------------------------------------------------------------

def build_rules_url(year: int) -> str:
    return f"https://iscottb122.github.io/{year}/index.html"


def build_data_url(year: int) -> str:
    blob_url = f"https://github.com/iscottb122/iscottb122.github.io/blob/master/{year}/data.txt"
    return blob_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _year_cache_dir(year: int) -> Path:
    cache_path = CACHE_DIR / str(year)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def _clear_year_cache(year: int) -> None:
    cache_path = CACHE_DIR / str(year)
    if cache_path.exists():
        shutil.rmtree(cache_path)
        logger.info("Cleared cache for year %d", year)


def _read_cached_text(year: int, filename: str) -> str | None:
    cache_path = CACHE_DIR / str(year) / filename
    if cache_path.exists():
        try:
            return cache_path.read_text(encoding="utf-8")
        except Exception:
            logger.exception("Failed reading cached file %s", cache_path)
    return None


def _write_cache_text(year: int, filename: str, content: str) -> None:
    cache_path = _year_cache_dir(year) / filename
    try:
        cache_path.write_text(content, encoding="utf-8")
        logger.info("Wrote cache: %s", cache_path)
    except Exception:
        logger.exception("Failed writing cache file %s", cache_path)


# ---------------------------------------------------------------------------
# Text loading
# ---------------------------------------------------------------------------

def load_text_from_source(source: str) -> str:
    source_path = Path(source)
    if source_path.exists():
        try:
            return source_path.read_text(encoding="utf-8")
        except Exception:
            logger.exception("Failed reading local source %s", source)
            return ""
    try:
        raw_url = source
        if "github.com" in source and "/blob/" in source:
            raw_url = source.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        response = requests.get(raw_url, timeout=10)
        if response.status_code == 200:
            return response.text
        logger.warning("HTTP fetch failed %s status=%s", raw_url, response.status_code)
    except Exception:
        logger.exception("Exception fetching source %s", source)
    return ""


# ---------------------------------------------------------------------------
# Rules parsing
# ---------------------------------------------------------------------------

def _normalize_position_code(pos_token: str) -> str:
    letter = pos_token.strip().lower()[:1]
    if letter == "g":
        return "gk"
    if letter == "d":
        return "df"
    if letter == "m":
        return "mf"
    return "st"


def parse_rules_table_from_html(html_text: str) -> pd.DataFrame | None:
    try:
        tables = re.findall(r"<table[\s\S]*?>[\s\S]*?<\/table>", html_text, re.IGNORECASE)
        for tbl in tables:
            rows = re.findall(r"<tr[\s\S]*?>[\s\S]*?<\/tr>", tbl, re.IGNORECASE)
            if not rows:
                continue
            parsed_rows = []
            for row_html in rows:
                cell_texts = re.findall(r"<t[dh][^>]*>([\s\S]*?)<\/t[dh]>", row_html, re.IGNORECASE)
                clean_cells = [re.sub(r"<[^>]+>", "", cell).strip() for cell in cell_texts]
                if clean_cells:
                    parsed_rows.append(clean_cells)
            if not parsed_rows:
                continue
            header = parsed_rows[0]
            if not any(re.search(r"event", h, re.IGNORECASE) for h in header):
                continue
            data_rows = parsed_rows[1:]
            if not data_rows:
                continue
            max_cols = max(len(row_cells) for row_cells in data_rows)
            col_names = header[:max_cols]
            records = []
            for row_cells in data_rows:
                padded = row_cells + [""] * (len(col_names) - len(row_cells))
                records.append(padded[: len(col_names)])
            data_df = pd.DataFrame(records, columns=col_names)
            data_df = data_df.set_index(data_df.columns[0])
            for col in data_df.columns:
                data_df[col] = (
                    data_df[col]
                    .astype(str)
                    .apply(lambda cell: re.search(r"(-?\d+)", cell).group(1) if re.search(r"(-?\d+)", cell) else "")
                )
                data_df[col] = pd.to_numeric(data_df[col], errors="coerce")
            return data_df
    except Exception:
        logger.exception("Error parsing HTML rules table")
    return None


def parse_rules_text_to_map(text: str) -> tuple[dict, pd.DataFrame | None]:
    rules_map: dict = {}
    rules_df: pd.DataFrame | None = None
    try:
        for line in text.splitlines():
            match = re.search(r"([A-Za-z _()]+)[\:\-\s]+(\-?\d+)\s*points?", line, re.IGNORECASE)
            if match:
                key = re.sub(r"\s+", " ", match.group(1).strip().lower())
                rules_map[key] = int(match.group(2))
    except Exception:
        logger.exception("Error parsing rules text")

    pos_tokens = {
        "gk": ["gk", "goalkeeper", "gkp"],
        "df": ["df", "defender", "def"],
        "mf": ["mf", "midfielder", "mid"],
        "st": ["st", "striker", "fwd", "forward"],
    }

    normalized: dict = {}
    for raw_key, val in rules_map.items():
        canonical = re.sub(r"\s+", " ", raw_key.replace("(", " ").replace(")", " ").strip().lower())
        normalized[canonical] = val
        for pos_code, aliases in pos_tokens.items():
            for alias in aliases:
                if re.search(rf"\b{re.escape(alias)}\b", canonical):
                    base = re.sub(r"\s+", " ", re.sub(rf"\b{re.escape(alias)}\b", "", canonical).strip())
                    if base:
                        normalized[f"{base} {pos_code}"] = val
                        normalized[f"{base} ({pos_code})"] = val
                        normalized[f"{pos_code} {base}"] = val
                    else:
                        normalized[canonical] = val

    try:
        plain_text = re.sub(r"<[^>]+>", " ", text)
        for line in plain_text.splitlines():
            line = line.strip()
            if not line:
                continue
            match = re.search(
                r"(?P<part1>[A-Za-z ]{2,30})[^0-9\n]{0,10}(?P<part2>gk|goalkeeper|gkp|df|defender|def|mf|midfielder|mid|st|striker|fwd|forward)[^0-9\n]{0,10}(?P<val>-?\d+)",
                line, re.IGNORECASE,
            )
            if not match:
                match = re.search(
                    r"(?P<pos>gk|goalkeeper|gkp|df|defender|def|mf|midfielder|mid|st|striker|fwd|forward)[^A-Za-z0-9]{0,10}(?P<evt>[A-Za-z ]{2,30})[^0-9\n]{0,10}(?P<val>-?\d+)",
                    line, re.IGNORECASE,
                )
            if match:
                groups = match.groupdict()
                if "evt" in groups:
                    evt = groups["evt"].strip().lower()
                    pos_token = groups["pos"].strip().lower()
                else:
                    evt = groups["part1"].strip().lower()
                    pos_token = groups["part2"].strip().lower()
                val = int(groups["val"])
                pos_code = _normalize_position_code(pos_token)
                evt = re.sub(r"\s+", " ", re.sub(r"[()\[\]]", " ", evt)).strip()
                if evt:
                    normalized[f"{evt} {pos_code}"] = val
                    normalized[f"{evt} ({pos_code})"] = val
    except Exception:
        logger.exception("Error extracting position-specific rules")

    try:
        rules_df = parse_rules_table_from_html(text)
        if rules_df is not None and not rules_df.empty:
            for evt_idx in rules_df.index:
                for col in rules_df.columns:
                    val = rules_df.at[evt_idx, col]
                    if pd.isna(val):
                        continue
                    ev_norm = re.sub(r"\s+", " ", str(evt_idx)).strip().lower()
                    pos_code = _normalize_position_code(col.strip())
                    normalized[f"{ev_norm} {pos_code}"] = int(val)
                    normalized[f"{ev_norm} ({pos_code})"] = int(val)
            logger.info("Extracted rules table with shape %s", rules_df.shape)
    except Exception:
        logger.exception("Error processing rules table")

    logger.info("Parsed rules keys (first 20): %s", list(normalized.keys())[:20])
    return normalized, rules_df


def _render_page_with_playwright(url: str, rules_tab_text: str = "Rules", timeout: int = 10) -> str | None:
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        logger.info("Playwright not available; skipping render")
        return None
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=timeout * 1000)
            try:
                tab_elem = page.query_selector(f"text=/\\b{rules_tab_text}\\b/i")
                if tab_elem:
                    tab_elem.click()
                    page.wait_for_timeout(500)
            except Exception:
                pass
            rendered_html = page.content()
            browser.close()
            return rendered_html
    except Exception:
        logger.exception("Error rendering page with Playwright")
        return None


def extract_transfer_datetimes(text: str) -> list:
    results = []
    try:
        matches = re.findall(
            r"(\d{1,2}:\d{2}\s*BST[,\s]+[A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)\s+[A-Za-z]+\s+\d{4})", text
        )
        if not matches:
            matches = re.findall(r"(\d{1,2}:\d{2}\s*BST[^\n,]{0,50}\d{4})", text)
        for match_str in matches:
            cleaned = re.sub(r"(st|nd|rd|th)", "", match_str)
            cleaned = cleaned.replace("BST", "").replace(",", " ").strip()
            parsed_dt = pd.to_datetime(cleaned, dayfirst=False, errors="coerce")
            if pd.isna(parsed_dt):
                continue
            dt_utc = (parsed_dt - pd.Timedelta(hours=1)).tz_localize("UTC")
            results.append(dt_utc)
    except Exception:
        logger.exception("Error extracting transfer datetimes")
    return results


def fetch_and_cache_rules(year: int, force_refresh: bool = False) -> tuple[dict, pd.DataFrame | None]:
    global _transfer_dates

    # Try cache first — but only accept it if the rules table was actually present
    if not force_refresh:
        cached_html = _read_cached_text(year, "rules.html")
        if cached_html:
            rules_map, rules_df = parse_rules_text_to_map(cached_html)
            if rules_df is not None and not rules_df.empty:
                logger.info("Using cached rules for year %d (%d entries)", year, len(rules_map))
                _transfer_dates = extract_transfer_datetimes(cached_html)
                return rules_map, rules_df
            logger.warning("Cached rules for year %d had no scoring table — re-fetching", year)

    # Check for a local rules table file (e.g. rules_table_2024.html placed in the project root)
    local_rules_file = Path(__file__).parent / f"rules_table_{year}.html"
    rules_map: dict = {}
    rules_df: pd.DataFrame | None = None
    best_source: str = ""

    if local_rules_file.exists():
        local_html = local_rules_file.read_text(encoding="utf-8")
        rules_map, rules_df = parse_rules_text_to_map(local_html)
        best_source = local_html
        if rules_df is not None and not rules_df.empty:
            logger.info("Loaded rules from local file %s (%d entries)", local_rules_file.name, len(rules_map))

    # Fetch remote URL; try static first, then Playwright if no table found
    url = build_rules_url(year)
    static_text = load_text_from_source(url)
    if static_text:
        remote_map, remote_df = parse_rules_text_to_map(static_text)
        if remote_df is not None and not remote_df.empty:
            # Remote static HTML has the table — prefer it
            rules_map, rules_df = remote_map, remote_df
            best_source = static_text
            logger.info("Loaded rules from static fetch of %s (%d entries)", url, len(rules_map))

    if rules_df is None or rules_df.empty:
        rendered_html = _render_page_with_playwright(url)
        if rendered_html:
            rendered_map, rendered_df = parse_rules_text_to_map(rendered_html)
            if rendered_df is not None and not rendered_df.empty:
                rules_map, rules_df = rendered_map, rendered_df
                best_source = rendered_html
                logger.info("Loaded rules via Playwright for year %d (%d entries)", year, len(rules_map))

    # Only cache if we actually have a valid scoring table
    if best_source and rules_df is not None and not rules_df.empty:
        _write_cache_text(year, "rules.html", best_source)

    _transfer_dates = extract_transfer_datetimes(best_source)
    if _transfer_dates:
        logger.info("Extracted %d transfer datetimes", len(_transfer_dates))

    if not rules_map:
        logger.warning("No rules loaded for year %d — all points will be 0", year)
    else:
        sample = list(rules_map.items())[:6]
        logger.info("Rules sample for year %d: %s", year, sample)

    return rules_map, rules_df


# ---------------------------------------------------------------------------
# ISO 3166 / Flag images
# ---------------------------------------------------------------------------

def fetch_iso3166_country_codes(url: str = ISO_3166_WIKIPEDIA_URL) -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    tables = pd.read_html(io.StringIO(response.text), header=0)
    iso_df = tables[0].iloc[1:].reset_index(drop=True)
    result = iso_df.iloc[:, [0, 2]].copy()
    result.columns = ["country_name", "alpha_2"]
    footnote_pattern = re.compile(r"\[[^\]]*\]")
    for col in result.columns:
        result[col] = result[col].astype(str).str.replace(footnote_pattern, "", regex=True).str.strip()
    result = result[result["alpha_2"].str.fullmatch(r"[A-Z]{2}")].reset_index(drop=True)
    return result


def download_flag_images(country_codes: list) -> None:
    ASSETS_DIR.mkdir(exist_ok=True)
    for alpha2 in country_codes:
        alpha2_upper = str(alpha2).strip().upper()
        if not alpha2_upper:
            continue
        flag_file = ASSETS_DIR / f"flag_{alpha2_upper}.png"
        if flag_file.exists():
            with flag_file.open("rb") as flag_fp:
                if flag_fp.read(8) == PNG_HEADER_BYTES:
                    continue
        try:
            image_response = requests.get(f"{FLAG_CDN_BASE_URL}{alpha2_upper.lower()}.png", timeout=10)
            image_response.raise_for_status()
            if image_response.content[:8] != PNG_HEADER_BYTES:
                raise ValueError(f"Response for {alpha2_upper} is not a valid PNG")
            flag_file.write_bytes(image_response.content)
            logger.info("Downloaded flag for %s", alpha2_upper)
        except Exception as exc:
            logger.warning("Could not download flag for %s: %s", alpha2_upper, exc)


def get_country_flag_asset_path(alpha2_code: str) -> str | None:
    a2 = str(alpha2_code).strip().upper()
    for prefix in ("shirt", "flag"):
        file_path = ASSETS_DIR / f"{prefix}_{a2}.png"
        if file_path.exists():
            with file_path.open("rb") as img_fp:
                if img_fp.read(8) == PNG_HEADER_BYTES:
                    return f"/assets/{prefix}_{a2}.png"
    return None


# ---------------------------------------------------------------------------
# Data text parsing
# ---------------------------------------------------------------------------

def get_event_points(event_key: str, pos: str, rules_map: dict) -> int:
    if not event_key:
        return 0
    ek = event_key.strip().lower()
    pos_key = pos.strip().lower()
    for candidate in [f"{ek} {pos_key}", f"{ek} ({pos_key})", ek]:
        if candidate in rules_map:
            return int(rules_map[candidate])
    return 0


def parse_data_txt(text: str, _rules_map: dict) -> dict:
    managers: list = []
    players: dict = {}
    events: list = []

    raw_lines = [line.rstrip() for line in text.splitlines()]

    for line in raw_lines:
        if line.strip().lower().startswith("managers:"):
            parts = line.split(":", 1)[1]
            for mgr_name in re.split(r"[;,]", parts):
                cleaned = mgr_name.strip()
                if cleaned:
                    managers.append(cleaned)
            break

    if not managers:
        for line in raw_lines:
            match = re.match(r"^\s*manager\s*[:\-]?\s*(.+)$", line, re.IGNORECASE)
            if match:
                mgr_name = match.group(1).strip()
                if mgr_name:
                    managers.append(mgr_name)

    if not managers:
        logger.warning("No managers found in data.txt")

    current_country = None
    for line in raw_lines:
        country_match = re.match(r"^\s*country\s+([A-Za-z]{2}(?:-[A-Za-z]{2,3})?)\s*(.*)$", line, re.IGNORECASE)
        if country_match:
            code = country_match.group(1).upper()
            desc = country_match.group(2).strip() if country_match.group(2) else f"Country ({code})"
            current_country = code
            players[current_country] = {"code": current_country, "description": desc, "players": []}
            continue
        player_match = re.match(r"^\s*player\s+(?P<pos>\w+)\s+(?P<name>.+)$", line, re.IGNORECASE)
        if player_match and current_country:
            players[current_country]["players"].append({
                "position": player_match.group("pos").strip(),
                "name": player_match.group("name").strip(),
            })

    draft_picks: list = []
    flat_name_to_code: dict = {}
    for country_code, country_info in players.items():
        for player_entry in country_info.get("players", []):
            flat_name_to_code[player_entry["name"].lower()] = (country_code, player_entry["name"])

    def _resolve_pick_token_to_name(token: str) -> str | None:
        token = token.strip()
        if not token:
            return None
        if re.match(r"^[A-Za-z]{2}$", token):
            code = token.upper()
            country_info = players.get(code)
            if country_info and country_info.get("players"):
                return country_info["players"][0]["name"]
            return players.get(code, {}).get("description", code)
        lookup = flat_name_to_code.get(token.lower())
        if lookup:
            return lookup[1]
        candidates = [full_name for name_key, (_, full_name) in flat_name_to_code.items() if token.lower() in name_key]
        if len(candidates) == 1:
            return candidates[0]
        code_match = re.search(r"([A-Za-z]{2})", token)
        if code_match:
            code = code_match.group(1).upper()
            country_info = players.get(code)
            if country_info and country_info.get("players"):
                return country_info["players"][0]["name"]
            return players.get(code, {}).get("description", code)
        return token

    for line in raw_lines:
        draft_match = re.match(r"^\s*draft\b\s*(.*)$", line, re.IGNORECASE)
        if not draft_match:
            continue
        rest = draft_match.group(1).strip()
        if not rest:
            continue
        for pick_token in [part.strip() for part in re.split(r"[;,]", rest) if part.strip()]:
            resolved = _resolve_pick_token_to_name(pick_token)
            draft_picks.append(resolved or pick_token.strip())

    normalized_picks = []
    for token in draft_picks:
        if token is None:
            continue
        if re.match(r"^[A-Za-z]{2}$", str(token)):
            code = str(token).upper()
            if code in players and players[code].get("players"):
                normalized_picks.append(players[code]["players"][0]["name"])
            else:
                normalized_picks.append(str(token))
            continue
        resolved = _resolve_pick_token_to_name(str(token))
        if resolved:
            normalized_picks.append(resolved)
        else:
            logger.warning("Unresolved draft pick preserved: %s", token)
            normalized_picks.append(str(token))
    draft_picks = normalized_picks

    teams_initial: dict = {mgr: [] for mgr in managers}
    manager_count = max(1, len(managers))
    for pick_idx, pick_code in enumerate(draft_picks):
        round_num = pick_idx // manager_count
        position_in_round = pick_idx % manager_count
        if round_num % 2 == 0:
            mgr_idx = position_in_round
        else:
            mgr_idx = manager_count - 1 - position_in_round
        mgr = managers[mgr_idx]
        teams_initial[mgr].append(pick_code)

    draft_complete = all(len(team) >= 11 for team in teams_initial.values()) if managers else False
    if managers and not draft_complete:
        logger.warning("Draft did not produce 11 players per manager: %s", {k: len(v) for k, v in teams_initial.items()})

    teams_by_gw: list = []
    current_teams: dict = {mgr: list(teams_initial.get(mgr, [])) for mgr in managers}
    teams_by_gw.append({mgr: list(current_teams.get(mgr, [])) for mgr in managers})

    def _resolve_token_to_player(token: str) -> str | None:
        if not token:
            return None
        cleaned_token = token.strip()
        lookup = flat_name_to_code.get(cleaned_token.lower())
        if lookup:
            return lookup[1]
        for name_key, (_, full_name) in flat_name_to_code.items():
            if cleaned_token.lower() in name_key:
                return full_name
        code_match = re.search(r"([A-Za-z]{2})", cleaned_token)
        if code_match:
            code = code_match.group(1).upper()
            country_info = players.get(code)
            if country_info and country_info.get("players"):
                return country_info["players"][0]["name"]
            return None
        return cleaned_token

    last_draft_idx = -1
    for line_idx, line_text in enumerate(raw_lines):
        if re.match(r"^\s*draft\b", line_text, re.IGNORECASE):
            last_draft_idx = line_idx
    remaining_lines = raw_lines[last_draft_idx + 1:]

    gw = 0
    appended_gws: set = {0}
    pending_out_mgr = None

    event_token_map = {str(val).strip().lower(): key.strip().lower() for key, val in EVENT_MAP.items()}

    for line in remaining_lines:
        stripped = line.strip()
        if not stripped:
            continue

        if re.match(r"^\s*transfers?\b", stripped, re.IGNORECASE):
            gw += 1
            pending_out_mgr = None
            continue

        if re.match(r"^out\b", stripped, re.IGNORECASE):
            token = re.sub(r"^out\b", "", stripped, flags=re.IGNORECASE).strip(" :,-")
            out_name = _resolve_token_to_player(token)
            assigned_mgr = None
            if out_name:
                for mgr, team in current_teams.items():
                    for member in list(team):
                        mem_name = member.get("name") if isinstance(member, dict) else str(member)
                        if mem_name and out_name.strip().lower() == mem_name.strip().lower():
                            assigned_mgr = mgr
                            break
                    if assigned_mgr:
                        break
            if assigned_mgr:
                current_teams[assigned_mgr] = [
                    member for member in current_teams[assigned_mgr]
                    if not (
                        (isinstance(member, dict) and member.get("name") and member.get("name").strip().lower() == out_name.strip().lower())
                        or (isinstance(member, str) and str(member).strip().lower() == out_name.strip().lower())
                    )
                ]
                pending_out_mgr = assigned_mgr
            else:
                logger.warning("OUT token not matched to any manager team: %s", token)
            if gw > 0 and gw not in appended_gws:
                teams_by_gw.append({mgr: list(current_teams.get(mgr, [])) for mgr in managers})
                appended_gws.add(gw)
            continue

        if re.match(r"^in\b", stripped, re.IGNORECASE):
            token = re.sub(r"^in\b", "", stripped, flags=re.IGNORECASE).strip(" :,-")
            in_name = _resolve_token_to_player(token)
            if pending_out_mgr:
                current_teams[pending_out_mgr].append(in_name if in_name else token)
                pending_out_mgr = None
            else:
                logger.warning("IN encountered with no pending OUT: %s", token)
            if gw > 0 and gw not in appended_gws:
                teams_by_gw.append({mgr: list(current_teams.get(mgr, [])) for mgr in managers})
                appended_gws.add(gw)
            continue

        first_word = stripped.split(None, 1)[0].lower()
        if first_word in event_token_map:
            rule = event_token_map[first_word]
            rest = stripped[len(first_word):].strip(" :,-")
            player_name = _resolve_token_to_player(rest) or rest
            events.append({"player_name": player_name, "Event": rule, "Gameweek": gw})
            if gw > 0 and gw not in appended_gws:
                teams_by_gw.append({mgr: list(current_teams.get(mgr, [])) for mgr in managers})
                appended_gws.add(gw)

    for gw_idx, snapshot in enumerate(teams_by_gw):
        for mgr, team in snapshot.items():
            if len(team) != 11:
                logger.warning("Gameweek %d: manager %s has %d players (expected 11)", gw_idx, mgr, len(team))

    return {"managers": managers, "players": players, "events": events, "teams_by_gw": teams_by_gw}


# ---------------------------------------------------------------------------
# Merged dataframe builder
# ---------------------------------------------------------------------------

def build_merged_dataframe(
    managers: list,
    players: dict,
    events: list,
    teams_by_gw_snapshots: list,
    parsed_rules: dict,
) -> pd.DataFrame:
    if not managers:
        return pd.DataFrame(columns=["Manager", "player", "gameweek", "position", "country", "event", "points"])

    teams_data: dict = {}
    for gw_idx, snapshot in enumerate(teams_by_gw_snapshots):
        col = f"GW{gw_idx}"
        for mgr, team in snapshot.items():
            teams_data.setdefault(mgr, {})[col] = list(team)

    wide_df = pd.DataFrame.from_dict(teams_data, orient="index").sort_index()

    team_rows = []
    for mgr in wide_df.index:
        for col in wide_df.columns:
            gw_num = None
            if isinstance(col, str) and col.startswith("GW"):
                try:
                    gw_num = int(col[2:])
                except Exception:
                    pass
            cell = wide_df.at[mgr, col]
            if cell is None or (isinstance(cell, float) and pd.isna(cell)):
                continue
            if isinstance(cell, (list, tuple, set)):
                player_list = list(cell)
            elif hasattr(cell, "__iter__") and not isinstance(cell, str):
                try:
                    player_list = list(cell)
                except Exception:
                    player_list = [cell]
            else:
                player_list = [cell]
            for player_entry in player_list:
                if player_entry is None or (isinstance(player_entry, float) and pd.isna(player_entry)):
                    continue
                mgr_display = mgr
                if isinstance(mgr, str) and mgr.strip():
                    name_parts = mgr.strip().split(None, 1)
                    if len(name_parts) == 2:
                        mgr_display = name_parts[1]
                team_rows.append({"Manager": mgr_display, "player": player_entry, "gameweek": gw_num})

    teams_df = pd.DataFrame(team_rows, columns=["Manager", "player", "gameweek"]) if team_rows else pd.DataFrame(columns=["Manager", "player", "gameweek"])

    event_rows = []
    for event_entry in events:
        pname = event_entry.get("player_name") or event_entry.get("player") or ""
        evt = event_entry.get("Event") or event_entry.get("event") or ""
        gw_num = event_entry.get("Gameweek") if event_entry.get("Gameweek") is not None else event_entry.get("gameweek", 0)
        event_rows.append({"player_name": pname, "event": evt, "gameweek": gw_num})
    events_df = pd.DataFrame(event_rows, columns=["player_name", "event", "gameweek"]) if event_rows else pd.DataFrame(columns=["player_name", "event", "gameweek"])

    player_rows = []
    for country, country_info in players.items():
        for player_entry in country_info.get("players", []):
            player_rows.append({
                "name": player_entry.get("name"),
                "country": country,
                "position": player_entry.get("position"),
            })
    players_df = pd.DataFrame(player_rows, columns=["name", "country", "position"]) if player_rows else pd.DataFrame(columns=["name", "country", "position"])

    if teams_df.empty:
        return pd.DataFrame(columns=["Manager", "player", "gameweek", "position", "country", "event", "points"])

    merged = teams_df.merge(events_df, left_on=["player", "gameweek"], right_on=["player_name", "gameweek"], how="left")
    merged = merged.merge(players_df, left_on="player", right_on="name", how="left")

    def _row_points(row_series):
        ev = row_series.get("event") or ""
        pos_raw = row_series.get("position")
        # Guard against NaN explicitly — NaN is truthy in Python so `pos_raw or ""` would
        # pass NaN through, causing _normalize_position_code("nan") → "st" for all positions.
        if pos_raw is None or (isinstance(pos_raw, float) and pd.isna(pos_raw)) or str(pos_raw).strip() == "":
            pos_code = ""
        else:
            pos_code = _normalize_position_code(str(pos_raw))
        try:
            return int(get_event_points(ev, pos_code, parsed_rules))
        except Exception:
            return 0

    merged["points"] = merged.apply(_row_points, axis=1)
    merged["event"] = merged["event"].fillna("")
    out_cols = ["Manager", "player", "gameweek", "position", "country", "event", "points"]
    return merged[out_cols].copy()


def _build_ui_dataframe(core_df: pd.DataFrame) -> pd.DataFrame:
    if core_df.empty:
        return pd.DataFrame()
    data_df = core_df.copy()
    data_df["User"] = data_df["Manager"]
    data_df["Gameweek"] = data_df["gameweek"]
    data_df["total_points"] = data_df["points"]
    data_df["Player"] = data_df["player"]
    data_df["name"] = data_df["player"]
    data_df["PlayerID"] = data_df["player"].apply(lambda x: abs(hash(str(x))) % (10**8))
    data_df["team_code"] = data_df.get("country", pd.Series("", index=data_df.index)).fillna("").astype(str)
    data_df["team_short_name"] = data_df["team_code"]

    try:
        iso_df = fetch_iso3166_country_codes()
        data_df = data_df.merge(iso_df, left_on="team_code", right_on="alpha_2", how="left")
    except Exception:
        logger.exception("Failed ISO merge in _build_ui_dataframe")
        data_df["country_name"] = None
        data_df["alpha_2"] = data_df["team_code"]

    missing_mask = data_df["country_name"].isna()
    if missing_mask.any():
        subdiv_upper = data_df.loc[missing_mask, "team_code"].str.upper()
        data_df.loc[missing_mask, "country_name"] = subdiv_upper.map(GB_SUBDIVISION_CODES)
        data_df.loc[missing_mask, "alpha_2"] = subdiv_upper

    _ev = data_df["event"].fillna("").str.lower()
    data_df["goals_scored"] = (_ev == "goal (excl. shootout)").astype(int)
    data_df["assists"] = (_ev == "assist").astype(int)
    data_df["clean_sheets"] = (_ev == "clean sheet (full time *)").astype(int)
    data_df["penalty_goals"] = (_ev == "scored penalty (shootout)").astype(int)
    data_df["own_goals"] = (_ev == "own goal").astype(int)
    data_df["red_cards"] = (_ev == "red card").astype(int)
    data_df["penalty_saves"] = (_ev == "penalty save (incl. shootout)").astype(int)
    data_df["missed_penalties"] = (_ev == "missed/saved penalty (incl. shootout)").astype(int)
    data_df["Position"] = 1

    grp_cols = ["User", "Player", "Gameweek", "team_code", "alpha_2", "country_name", "team_short_name", "name"]
    if "position" in data_df.columns:
        grp_cols.append("position")

    agg_df = data_df.groupby(grp_cols, dropna=False).agg({
        "total_points": "sum",
        "goals_scored": "sum",
        "assists": "sum",
        "clean_sheets": "sum",
        "penalty_goals": "sum",
        "own_goals": "sum",
        "red_cards": "sum",
        "penalty_saves": "sum",
        "missed_penalties": "sum",
    }).reset_index()

    if "Position" not in agg_df.columns:
        agg_df["Position"] = 1
    agg_df["player_name"] = agg_df["name"]
    return agg_df


# ---------------------------------------------------------------------------
# Year data loader (main entry point)
# ---------------------------------------------------------------------------

def load_year_data(year: int, force_cache_refresh: bool = False) -> None:
    global _current_year, _merged_df, _user_options, _gameweek_options, _transfer_dates

    logger.info("Loading data for year %d (force_refresh=%s)", year, force_cache_refresh)

    if force_cache_refresh:
        _clear_year_cache(year)

    try:
        parsed_rules, _rules_df = fetch_and_cache_rules(year, force_refresh=force_cache_refresh)
    except Exception:
        logger.exception("Failed to fetch rules for year %d", year)
        parsed_rules = {}

    cached_data = None if force_cache_refresh else _read_cached_text(year, "data.txt")
    if cached_data:
        logger.info("Using cached data.txt for year %d", year)
        raw_data_text = cached_data
    else:
        data_url = build_data_url(year)
        raw_data_text = load_text_from_source(data_url)
        if raw_data_text:
            _write_cache_text(year, "data.txt", raw_data_text)

    try:
        parsed = parse_data_txt(raw_data_text or "", parsed_rules)
    except Exception:
        logger.exception("Failed to parse data.txt for year %d", year)
        parsed = {"managers": [], "players": {}, "events": [], "teams_by_gw": []}

    managers = parsed.get("managers", [])
    players = parsed.get("players", {})
    events = parsed.get("events", [])
    teams_by_gw_snapshots = parsed.get("teams_by_gw", [])

    try:
        core_df = build_merged_dataframe(managers, players, events, teams_by_gw_snapshots, parsed_rules)
    except Exception:
        logger.exception("Failed to build merged dataframe")
        core_df = pd.DataFrame()

    try:
        ui_df = _build_ui_dataframe(core_df)
    except Exception:
        logger.exception("Failed to build UI dataframe")
        ui_df = pd.DataFrame()

    if not ui_df.empty and "alpha_2" in ui_df.columns:
        alpha2_codes = ui_df["alpha_2"].dropna().unique().tolist()
        try:
            download_flag_images(alpha2_codes)
        except Exception:
            logger.warning("Flag image download failed")

    _merged_df = ui_df
    _current_year = year
    _user_options = sorted(_merged_df["User"].unique().tolist()) if not _merged_df.empty else []
    if not _merged_df.empty and "Gameweek" in _merged_df.columns:
        _gameweek_options = ["All"] + sorted(_merged_df["Gameweek"].unique().tolist(), reverse=True)
    else:
        _gameweek_options = ["All"]

    logger.info(
        "Year %d loaded: %d users, %d gameweeks, %d rows",
        year, len(_user_options), len(_gameweek_options) - 1, len(_merged_df),
    )


# ---------------------------------------------------------------------------
# Perform initial load at module level
# ---------------------------------------------------------------------------
try:
    load_year_data(DEFAULT_YEAR)
except Exception:
    logger.exception("Error during module-level data load")

# ---------------------------------------------------------------------------
# Player stats helpers
# ---------------------------------------------------------------------------

def build_player_stats_summary(data_df: pd.DataFrame, pos_filter: str = "all") -> pd.DataFrame:
    if data_df.empty:
        return pd.DataFrame(columns=["Player", "Position", "Country", "Points", "Appearances",
                                     "Goals", "Pen. Goals", "Assists", "Clean Sheets",
                                     "Pen. Saves", "Own Goals", "Red Cards", "Missed Pens."])

    working = data_df.copy()
    if pos_filter != "all" and "position" in working.columns:
        mask = working["position"].fillna("").str.upper().str.startswith(pos_filter.upper())
        working = working[mask]

    grp_cols = ["player_name"]
    if "position" in working.columns:
        grp_cols.append("position")
    if "country_name" in working.columns:
        grp_cols.append("country_name")

    metric_cols = {col: (col, "sum") for col in
                   ["goals_scored", "penalty_goals", "assists", "clean_sheets",
                    "penalty_saves", "own_goals", "red_cards", "missed_penalties"]
                   if col in working.columns}

    agg_kwargs = {
        "Points": ("total_points", "sum"),
        "Appearances": ("Gameweek", "nunique"),
        **{k: v for k, v in metric_cols.items()},
    }
    agg_df = working.groupby(grp_cols, dropna=False).agg(**agg_kwargs).reset_index()

    rename_map = {
        "player_name": "Player",
        "position": "Position",
        "country_name": "Country",
        "goals_scored": "Goals",
        "penalty_goals": "Pen. Goals",
        "assists": "Assists",
        "clean_sheets": "Clean Sheets",
        "penalty_saves": "Pen. Saves",
        "own_goals": "Own Goals",
        "red_cards": "Red Cards",
        "missed_penalties": "Missed Pens.",
    }
    agg_df = agg_df.rename(columns=rename_map)
    return agg_df.sort_values("Points", ascending=False).reset_index(drop=True)


def build_player_gw_breakdown(data_df: pd.DataFrame, player_name: str) -> pd.DataFrame:
    player_rows = data_df[data_df["player_name"] == player_name].copy()

    metric_agg = {col: (col, "sum") for col in
                  ["goals_scored", "penalty_goals", "assists", "clean_sheets",
                   "penalty_saves", "own_goals", "red_cards", "missed_penalties"]
                  if col in player_rows.columns}

    gw_agg = player_rows.groupby("Gameweek", dropna=False).agg(
        Points=("total_points", "sum"),
        **metric_agg,
    ).reset_index()

    rename_map = {
        "goals_scored": "Goals",
        "penalty_goals": "Pen. Goals",
        "assists": "Assists",
        "clean_sheets": "Clean Sheets",
        "penalty_saves": "Pen. Saves",
        "own_goals": "Own Goals",
        "red_cards": "Red Cards",
        "missed_penalties": "Missed Pens.",
    }
    gw_agg = gw_agg.rename(columns=rename_map)
    return gw_agg.sort_values("Gameweek").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Chart/table helpers
# ---------------------------------------------------------------------------

def aggregate_points_by_filter(data_df: pd.DataFrame, filter_type: str, column_name: str) -> tuple:
    starters = data_df[data_df["Position"] <= 11]
    if filter_type == "all":
        agg = starters.groupby(["User", "Gameweek"])[column_name].sum().reset_index()
        title = column_name.replace("_", " ").capitalize()
    elif filter_type in ("GK", "DF", "MF", "ST", "FW") and "position" in data_df.columns:
        filtered = starters[starters["position"].str.upper() == filter_type.upper()]
        agg = filtered.groupby(["User", "Gameweek"])[column_name].sum().reset_index()
        title = f"{column_name.replace('_', ' ').capitalize()} — {filter_type}"
    else:
        agg = starters.groupby(["User", "Gameweek"])[column_name].sum().reset_index()
        title = column_name.replace("_", " ").capitalize()
    pivot = agg.pivot(index="Gameweek", columns="User", values=column_name)
    return pivot, title


def build_top10_player_pie_chart(data_df: pd.DataFrame) -> go.Figure:
    top_players = data_df.groupby("player_name")["total_points"].sum().nlargest(10)
    fig = px.pie(
        names=top_players.index,
        values=top_players.values,
        title="Top 10 Player Contributions",
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.RdBu,
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(
        margin=dict(t=40, b=20, l=20, r=20),
        showlegend=False,
        plot_bgcolor=COLOUR_TRANSPARENT,
        paper_bgcolor=COLOUR_DARK_GREY,
        font=dict(color=COLOUR_WHITE),
    )
    return fig


def build_ranked_colour_table(standings_df: pd.DataFrame):
    num_cols = len(standings_df.columns)
    scale_size = str(max(3, min(num_cols, 11)))
    colours = cl.scales[scale_size]["div"]["RdBu"]
    style_data_conditional = []
    for col in standings_df.columns:
        for col_idx, colour in enumerate(colours):
            style_data_conditional.append({
                "if": {"column_id": col, "filter_query": f"{{{col}}} = {col_idx + 1}"},
                "backgroundColor": colour,
                "color": COLOUR_BLACK,
                "fontWeight": "bold",
            })
    style_data_conditional.append({
        "if": {
            "column_id": standings_df.reset_index().columns[0],
            "filter_query": "1 != 0",
        },
        "backgroundColor": COLOUR_DARK_GREY,
        "color": COLOUR_WHITE,
        "fontWeight": "bold",
    })
    return dash_table.DataTable(
        data=standings_df.sort_index(ascending=False).reset_index().to_dict("records"),
        columns=[{"name": col_name, "id": col_name} for col_name in standings_df.reset_index().columns],
        style_data_conditional=style_data_conditional,
        style_cell={"textAlign": "center"},
        style_header={"backgroundColor": COLOUR_DARK_GREY, "fontWeight": "bold"},
    )


def discrete_background_color_bins(data_df: pd.DataFrame, n_bins: int = 5, columns="all", scale: str = "Blues"):
    import colorlover
    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    if columns == "all":
        if "id" in data_df:
            df_numeric = data_df.select_dtypes("number").drop(["id"], axis=1)
        else:
            df_numeric = data_df.select_dtypes("number")
    else:
        df_numeric = data_df[columns]
    df_max = df_numeric.max().max()
    df_min = df_numeric.min().min()
    ranges = [((df_max - df_min) * b) + df_min for b in bounds]
    styles = []
    legend_items = []
    for bin_idx in range(1, len(bounds)):
        min_bound = ranges[bin_idx - 1]
        max_bound = ranges[bin_idx]
        background_colour = colorlover.scales[str(n_bins)]["seq"][scale][bin_idx - 1]
        text_colour = COLOUR_WHITE if bin_idx > len(bounds) / 2.0 else "inherit"
        for col_name in df_numeric:
            styles.append({
                "if": {
                    "filter_query": (
                        f"{{{col_name}}} >= {min_bound}"
                        + (f" && {{{col_name}}} < {max_bound}" if bin_idx < len(bounds) - 1 else "")
                    ),
                    "column_id": col_name,
                },
                "backgroundColor": background_colour,
                "color": text_colour,
            })
        legend_items.append(
            html.Div(style={"display": "inline-block", "width": "60px"}, children=[
                html.Div(style={"backgroundColor": background_colour, "borderLeft": "1px rgb(50, 50, 50) solid", "height": "10px"}),
                html.Small(round(min_bound, 2), style={"paddingLeft": "2px"}),
            ])
        )
    return styles, html.Div(legend_items, style={"padding": "5px 0 5px 0"})


# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
app.layout = html.Div(
    style={"backgroundColor": COLOUR_DARK_GREY, "minHeight": "100vh"},
    children=[
        html.Div(
            style={
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "space-between",
                "padding": "10px 20px",
                "backgroundColor": COLOUR_BLACK,
            },
            children=[
                html.H2("World Cup Draft", style={"color": COLOUR_WHITE, "margin": 0}),
                html.Div(
                    style={"display": "flex", "alignItems": "center", "gap": "10px"},
                    children=[
                        html.Label("Year:", style={"color": COLOUR_WHITE, "margin": 0}),
                        dcc.Dropdown(
                            id="year-dropdown",
                            options=[{"label": str(yr), "value": yr} for yr in AVAILABLE_YEARS],
                            value=DEFAULT_YEAR,
                            clearable=False,
                            style={"width": "120px", "color": COLOUR_BLACK},
                        ),
                        html.Button(
                            "Refresh Cache",
                            id="refresh-cache-btn",
                            n_clicks=0,
                            disabled=False,
                            style={
                                "backgroundColor": "#555",
                                "color": COLOUR_WHITE,
                                "border": "none",
                                "padding": "6px 14px",
                                "cursor": "pointer",
                                "borderRadius": "4px",
                            },
                        ),
                        html.Span(id="refresh-status", style={"color": "#aaa", "fontSize": "12px"}),
                    ],
                ),
            ],
        ),
        dcc.Store(id="data-reload-trigger", data=0),
        dcc.Tabs(
            id="main-tabs",
            value="all-users-view",
            children=[
                dcc.Tab(label="All Users Stats", value="all-users-view"),
                dcc.Tab(label="User Stats", value="single-user-view"),
                dcc.Tab(label="Player Stats", value="player-stats-view"),
            ],
        ),
        html.Div(id="tab-content", style={"padding": "20px"}),
    ],
)


# ---------------------------------------------------------------------------
# Disable refresh button for previous (completed) years
# ---------------------------------------------------------------------------
@app.callback(
    Output("refresh-cache-btn", "disabled"),
    Output("refresh-cache-btn", "style"),
    Input("year-dropdown", "value"),
)
def toggle_refresh_button(selected_year):
    is_previous_year = selected_year != DEFAULT_YEAR
    base_style = {
        "border": "none",
        "padding": "6px 14px",
        "borderRadius": "4px",
    }
    if is_previous_year:
        return True, {**base_style, "backgroundColor": "#333", "color": "#666", "cursor": "not-allowed"}
    return False, {**base_style, "backgroundColor": "#555", "color": COLOUR_WHITE, "cursor": "pointer"}


# ---------------------------------------------------------------------------
# Data reload callback (year change + refresh button)
# ---------------------------------------------------------------------------
@app.callback(
    Output("data-reload-trigger", "data"),
    Output("refresh-status", "children"),
    Input("year-dropdown", "value"),
    Input("refresh-cache-btn", "n_clicks"),
    State("data-reload-trigger", "data"),
    prevent_initial_call=True,
)
def handle_data_reload(selected_year, _n_clicks, current_trigger):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""
    force_refresh = trigger_id == "refresh-cache-btn" and selected_year == DEFAULT_YEAR
    try:
        load_year_data(selected_year, force_cache_refresh=force_refresh)
        status_msg = f"Loaded {selected_year}" + (" (refreshed)" if force_refresh else "")
    except Exception:
        logger.exception("Error reloading data for year %d", selected_year)
        status_msg = f"Error loading {selected_year}"
    return (current_trigger or 0) + 1, status_msg


# ---------------------------------------------------------------------------
# Main tab router
# ---------------------------------------------------------------------------
@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "value"),
    Input("data-reload-trigger", "data"),
)
def render_main_tab(tab, _reload_trigger):
    if tab == "all-users-view":
        return html.Div([
            html.H4("Performance Stats (All Gameweeks)", style={"color": COLOUR_WHITE}),
            dcc.Tabs(
                id="stats-subtabs",
                value="standings-tab",
                children=[
                    dcc.Tab(label="Standings", value="standings-tab"),
                    dcc.Tab(label="Performance", value="performance-tab"),
                    dcc.Tab(label="Squad", value="squad-tab"),
                ],
            ),
            html.Div(id="stats-subtab-content", style={"marginTop": "20px"}),
        ])
    elif tab == "single-user-view":
        user_opts = _user_options
        gw_opts = _gameweek_options
        return html.Div([
            html.Div([
                html.Label("Select User:", style={"color": COLOUR_WHITE}),
                dcc.Dropdown(
                    id="user-dropdown",
                    options=[{"label": str(u), "value": u} for u in user_opts],
                    value=user_opts[0] if user_opts else None,
                    clearable=False,
                ),
            ], style={"width": "48%", "display": "inline-block"}),
            html.Div([
                html.Label("Select Gameweek:", style={"color": COLOUR_WHITE}),
                dcc.Dropdown(
                    id="gameweek-dropdown",
                    options=[{"label": f"GW {gw}" if gw != "All" else "All", "value": gw} for gw in gw_opts],
                    value=gw_opts[0],
                    clearable=False,
                ),
            ], style={"width": "48%", "display": "inline-block", "float": "right"}),
            html.Br(), html.Br(),
            html.Div([
                html.Div([
                    html.H4("Team", style={"color": COLOUR_WHITE}),
                    dcc.Graph(id="pitch-graph"),
                ], style={"width": "50%", "paddingLeft": "20px", "paddingRight": "20px",
                           "display": "inline-block", "verticalAlign": "top"}),
                html.Div([
                    html.H4("Gameweek Summary", style={"color": COLOUR_WHITE}),
                    html.Div(className="summary-panel", id="summary-panel"),
                    html.Br(),
                    html.Div([dcc.Graph(id="pie-chart")], className="graph-container"),
                ], style={"width": "48%", "display": "inline-block", "paddingLeft": "20px"}),
            ]),
        ])
    elif tab == "player-stats-view":
        summary_df = build_player_stats_summary(_merged_df, "all")
        summary_records = summary_df.to_dict("records") if not summary_df.empty else []
        return html.Div([
            html.H4("Player Stats", style={"color": COLOUR_WHITE}),
            html.Div([
                html.Label("Filter by Position:", style={"color": COLOUR_WHITE, "marginRight": "10px"}),
                dcc.Dropdown(
                    id="player-stats-pos-filter",
                    options=POSITION_FILTER_OPTIONS,
                    value="all",
                    clearable=False,
                    style={"width": "300px", "color": COLOUR_BLACK},
                ),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "15px"}),
            dash_table.DataTable(
                id="player-stats-summary-table",
                data=summary_records,
                columns=PLAYER_STATS_SUMMARY_COLUMNS,
                sort_action="native",
                filter_action="native",
                page_size=25,
                style_table={"overflowX": "auto"},
                style_header={
                    "backgroundColor": COLOUR_BLACK,
                    "color": COLOUR_WHITE,
                    "fontWeight": "bold",
                    "textAlign": "center",
                },
                style_cell={
                    "backgroundColor": COLOUR_DARK_GREY,
                    "color": COLOUR_WHITE,
                    "textAlign": "center",
                    "padding": "8px",
                },
                style_data={"border": "1px solid #555"},
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": "#404040"},
                    {
                        "if": {"state": "active"},
                        "backgroundColor": "#1a5276",
                        "border": "1px solid #5dade2",
                        "color": COLOUR_WHITE,
                    },
                    {
                        "if": {"state": "selected"},
                        "backgroundColor": "#1a5276",
                        "border": "1px solid #5dade2",
                        "color": COLOUR_WHITE,
                    },
                ],
            ),
            html.Div(
                id="player-stats-gw-container",
                style={"marginTop": "30px"},
                children=html.P(
                    "Click on a player row to view their per-gameweek breakdown.",
                    style={"color": "#888"},
                ),
            ),
        ])
    return html.Div()


# ---------------------------------------------------------------------------
# Stats sub-tabs
# ---------------------------------------------------------------------------
@app.callback(
    Output("stats-subtab-content", "children"),
    Input("stats-subtabs", "value"),
    Input("data-reload-trigger", "data"),
)
def render_stats_subtab(active_tab, _reload_trigger):
    if active_tab == "standings-tab":
        return html.Div([
            html.H5("Standings Overview", style={"color": COLOUR_WHITE}),
            dcc.RadioItems(
                className="dash-radio-items",
                id="standings-view-toggle",
                options=CHART_TABLE_TOGGLE_OPTIONS,
                value="table",
                labelStyle={"display": "inline-block", "marginRight": "10px"},
                style={"marginTop": "5px"},
            ),
            html.Br(),
            html.Div(id="standings-content"),
        ])
    elif active_tab == "performance-tab":
        return html.Div([
            html.Div([
                html.H5("Points per gameweek", style={"marginTop": "20px", "color": COLOUR_WHITE}),
                dcc.Dropdown(
                    id="performance-pos-filter",
                    options=POSITION_FILTER_OPTIONS,
                    value="all",
                    clearable=False,
                    style={"width": "300px", "marginTop": "5px"},
                ),
                dcc.RadioItems(
                    className="dash-radio-items",
                    id="performance-view-toggle",
                    options=CHART_TABLE_TOGGLE_OPTIONS,
                    value="table",
                    labelStyle={"display": "inline-block", "marginRight": "10px"},
                    style={"marginTop": "5px"},
                ),
                html.Br(),
                html.Div(id="performance-content"),
            ]),
            html.Br(),
            html.Div([
                html.H5("Cumulative points", style={"marginTop": "20px", "color": COLOUR_WHITE}),
                dcc.Dropdown(
                    id="cumulative-performance-pos-filter",
                    options=POSITION_FILTER_OPTIONS,
                    value="all",
                    clearable=False,
                    style={"width": "300px", "marginTop": "5px"},
                ),
                dcc.RadioItems(
                    className="dash-radio-items",
                    id="cumulative-performance-view-toggle",
                    options=CHART_TABLE_TOGGLE_OPTIONS,
                    value="chart",
                    labelStyle={"display": "inline-block", "marginRight": "10px", "marginTop": "5px"},
                    style={"marginTop": "5px"},
                ),
                html.Br(),
                html.Div(id="cumulative-performance-content"),
            ]),
            html.Br(),
            html.Div([
                html.H5("Performance spread", style={"marginTop": "20px", "color": COLOUR_WHITE}),
                dcc.RadioItems(
                    className="dash-radio-items",
                    id="performance-spread-view-toggle",
                    options=[{"label": "Mean", "value": "mean"}, {"label": "Spread", "value": "spread"}],
                    value="mean",
                    labelStyle={"display": "inline-block", "marginRight": "10px", "marginTop": "5px"},
                    style={"marginTop": "5px"},
                ),
                html.Div(id="performance-spread-content"),
            ]),
            html.Br(),
        ])
    elif active_tab == "squad-tab":
        return html.Div([
            html.Div([
                html.Div([
                    html.H5("Goals", style={"marginTop": "20px", "color": COLOUR_WHITE}),
                    dcc.Dropdown(
                        id="squad-goals-pos-filter",
                        options=POSITION_FILTER_OPTIONS,
                        value="all",
                        clearable=False,
                        style={"width": "300px", "marginTop": "5px"},
                    ),
                    dcc.RadioItems(
                        className="dash-radio-items",
                        id="squad-goals-view-toggle",
                        options=CHART_TABLE_TOGGLE_OPTIONS,
                        value="chart",
                        labelStyle={"display": "inline-block", "marginRight": "10px", "marginTop": "5px"},
                        style={"marginTop": "5px"},
                    ),
                    html.Br(),
                    html.Div(id="squad-goals-content"),
                ]),
                html.Br(),
                html.Div([
                    html.H5("Assists", style={"marginTop": "20px", "color": COLOUR_WHITE}),
                    dcc.Dropdown(
                        id="squad-assists-pos-filter",
                        options=POSITION_FILTER_OPTIONS,
                        value="all",
                        clearable=False,
                        style={"width": "300px", "marginTop": "5px"},
                    ),
                    dcc.RadioItems(
                        className="dash-radio-items",
                        id="squad-assists-view-toggle",
                        options=CHART_TABLE_TOGGLE_OPTIONS,
                        value="chart",
                        labelStyle={"display": "inline-block", "marginRight": "10px", "marginTop": "5px"},
                        style={"marginTop": "5px"},
                    ),
                    html.Br(),
                    html.Div(id="squad-assists-content"),
                ]),
                html.Br(),
                html.Div([
                    html.H5("Clean Sheets", style={"marginTop": "20px", "color": COLOUR_WHITE}),
                    dcc.Dropdown(
                        id="squad-clean-sheets-pos-filter",
                        options=POSITION_FILTER_OPTIONS,
                        value="all",
                        clearable=False,
                        style={"width": "300px", "marginTop": "5px"},
                    ),
                    dcc.RadioItems(
                        className="dash-radio-items",
                        id="squad-clean-sheets-view-toggle",
                        options=CHART_TABLE_TOGGLE_OPTIONS,
                        value="chart",
                        labelStyle={"display": "inline-block", "marginRight": "10px", "marginTop": "5px"},
                        style={"marginTop": "5px"},
                    ),
                    html.Br(),
                    html.Div(id="squad-clean-sheets-content"),
                ]),
                html.Br(),
            ]),
        ])
    return html.Div()


# ---------------------------------------------------------------------------
# Standings callback
# ---------------------------------------------------------------------------
@app.callback(
    Output("standings-content", "children"),
    Input("standings-view-toggle", "value"),
    Input("data-reload-trigger", "data"),
)
def update_standings_view(view_mode, _reload_trigger):
    if _merged_df.empty:
        return html.P("No data available.", style={"color": "#888"})
    standings_df = (
        _merged_df[_merged_df["Position"] <= 11]
        .groupby(["User", "Gameweek"])["total_points"].sum().reset_index()
    )
    standings_df = (
        standings_df.pivot(index="Gameweek", columns="User", values="total_points")
        .fillna(0).sort_index().cumsum()
        .sort_index(ascending=False)
        .rank(1, ascending=False, method="min").astype(int)
    )
    if view_mode == "chart":
        fig = px.line(standings_df, title="Position", labels={"User": ""})
        fig.update_layout(
            plot_bgcolor=COLOUR_TRANSPARENT,
            paper_bgcolor=COLOUR_TRANSPARENT,
            font=dict(color=COLOUR_WHITE),
            margin=dict(t=40, b=20, l=20, r=20),
        )
        return html.Div([dcc.Graph(figure=fig)], className="graph-container")
    return build_ranked_colour_table(standings_df)


# ---------------------------------------------------------------------------
# Single-user callbacks
# ---------------------------------------------------------------------------
@app.callback(
    Output("pitch-graph", "figure"),
    Output("pie-chart", "figure"),
    Output("summary-panel", "children"),
    Input("user-dropdown", "value"),
    Input("gameweek-dropdown", "value"),
    Input("data-reload-trigger", "data"),
)
def update_user_view(user, gw_selected, _reload_trigger):
    gw = _gameweek_options[1] if gw_selected == "All" and len(_gameweek_options) > 1 else gw_selected
    player_df = _merged_df[(_merged_df["User"] == user) & (_merged_df["Gameweek"] == gw)].copy()
    player_df["total_points"] = player_df["total_points"].fillna(0)

    pos_col = "position" if "position" in player_df.columns else None

    def assign_coordinates(players_sub_df: pd.DataFrame) -> dict:
        coords: dict = {}
        line_groups: dict[str | None, list] = {}
        for row_idx, row_data in players_sub_df.iterrows():
            raw_pos = str(row_data.get(pos_col, "") if pos_col else "").upper()
            line_key = next((k for k in POSITION_LINE_MAP if raw_pos.startswith(k[:2])), None)
            line_groups.setdefault(line_key, []).append(row_idx)
        for line_key, row_indices in line_groups.items():
            if line_key is None:
                y_coord, default_count = PITCH_HEIGHT / 10, len(row_indices)
            else:
                y_coord, default_count = POSITION_LINE_MAP[line_key]
            count = len(row_indices)
            if count == 1:
                x_positions = [PITCH_WIDTH / 2]
            else:
                margin = PITCH_WIDTH / max(default_count, count)
                x_positions = list(np.linspace(margin, PITCH_WIDTH - margin, count))
            for pos_idx, row_idx in enumerate(row_indices):
                coords[row_idx] = (x_positions[pos_idx], y_coord)
        return coords

    coords_map = assign_coordinates(player_df)
    player_df["_x"] = player_df.index.map(lambda row_idx: coords_map.get(row_idx, (PITCH_WIDTH / 2, PITCH_HEIGHT / 2))[0])
    player_df["_y"] = player_df.index.map(lambda row_idx: coords_map.get(row_idx, (PITCH_WIDTH / 2, PITCH_HEIGHT / 2))[1])

    pitch_shapes = []
    for stripe_idx in range(0, PITCH_HEIGHT, 10):
        stripe_colour = "#228B22" if (stripe_idx // 10) % 2 == 0 else "#32CD32"
        pitch_shapes.append(dict(
            type="rect", x0=0, x1=PITCH_WIDTH, y0=stripe_idx, y1=stripe_idx + 10,
            fillcolor=stripe_colour, line=dict(width=0), layer="below",
        ))
    pitch_shapes += [
        dict(type="line", x0=0, x1=PITCH_WIDTH, y0=95 * PITCH_HEIGHT / 100,
             y1=95 * PITCH_HEIGHT / 100, line=dict(color=COLOUR_WHITE, width=2)),
        dict(type="circle", x0=4 * PITCH_WIDTH / 10, x1=6 * PITCH_WIDTH / 10,
             y0=85 * PITCH_HEIGHT / 100, y1=110 * PITCH_HEIGHT / 100,
             line=dict(color=COLOUR_WHITE, width=2)),
        dict(type="rect", x0=3 * PITCH_WIDTH / 10, x1=7 * PITCH_WIDTH / 10,
             y0=0, y1=20 * PITCH_HEIGHT / 100, line=dict(color=COLOUR_WHITE, width=2)),
        dict(type="rect", x0=4 * PITCH_WIDTH / 10, x1=6 * PITCH_WIDTH / 10,
             y0=0, y1=10 * PITCH_HEIGHT / 100, line=dict(color=COLOUR_WHITE, width=2)),
    ]

    pitch_images = []
    pitch_annotations = []
    for _, row_data in player_df.iterrows():
        img_path = get_country_flag_asset_path(row_data.get("team_code", ""))
        if img_path:
            pitch_images.append(dict(
                source=img_path,
                x=row_data["_x"] - 5, y=row_data["_y"] - 2,
                xref="x", yref="y",
                sizex=10, sizey=10,
                xanchor="left", yanchor="bottom",
                layer="above",
            ))
        label = f"<b>{row_data['name']}</b><br><b>{int(row_data['total_points'])}</b>"
        pitch_annotations.append(dict(
            x=row_data["_x"], y=row_data["_y"] - 6.0,
            text=label, showarrow=False,
            font=dict(size=10, color=COLOUR_WHITE),
            bgcolor="#175717", align="center",
        ))

    pitch_fig = go.Figure()
    pitch_fig.update_layout(
        shapes=pitch_shapes,
        images=pitch_images,
        annotations=pitch_annotations,
        xaxis=dict(range=[0, PITCH_WIDTH], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-5, PITCH_HEIGHT], showgrid=False, zeroline=False, visible=False),
        plot_bgcolor=COLOUR_DARK_GREY,
        paper_bgcolor=COLOUR_TRANSPARENT,
        font=dict(color=COLOUR_WHITE),
        margin=dict(t=10, b=10, l=10, r=10),
        height=550,
    )

    stats_df = _merged_df[_merged_df["User"] == user]
    if gw_selected != "All":
        stats_df = stats_df[stats_df["Gameweek"] == gw]

    summary_panel = html.Ul([
        html.Li(f"Total Points: {stats_df['total_points'].sum()}"),
        html.Li(f"Goals Scored: {stats_df['goals_scored'].sum()}"),
        html.Li(f"Assists: {stats_df['assists'].sum()}"),
        html.Li(f"Clean Sheets: {stats_df['clean_sheets'].sum()}"),
    ], style={"color": COLOUR_WHITE})

    pie_fig = build_top10_player_pie_chart(stats_df)
    return pitch_fig, pie_fig, summary_panel


# ---------------------------------------------------------------------------
# Performance callbacks
# ---------------------------------------------------------------------------
@app.callback(
    Output("performance-content", "children"),
    Input("performance-pos-filter", "value"),
    Input("performance-view-toggle", "value"),
    Input("data-reload-trigger", "data"),
)
def update_performance_graph(filter_type, view_mode, _reload_trigger):
    agg, title = aggregate_points_by_filter(_merged_df, filter_type, "total_points")
    if view_mode == "chart":
        fig = px.line(agg, markers=True)
        fig.update_layout(title=title, xaxis_title="Gameweek", yaxis_title="Points",
                          plot_bgcolor=COLOUR_TRANSPARENT, paper_bgcolor=COLOUR_TRANSPARENT,
                          font=dict(color=COLOUR_WHITE))
        return html.Div([dcc.Graph(figure=fig)], className="graph-container")
    return build_ranked_colour_table(agg.sort_index(ascending=False).rank(1, ascending=False, method="min"))


@app.callback(
    Output("cumulative-performance-content", "children"),
    Input("cumulative-performance-pos-filter", "value"),
    Input("cumulative-performance-view-toggle", "value"),
    Input("data-reload-trigger", "data"),
)
def update_cumulative_performance_graph(filter_type, view_mode, _reload_trigger):
    agg, title = aggregate_points_by_filter(_merged_df, filter_type, "total_points")
    agg = agg.cumsum()
    if view_mode == "chart":
        fig = px.line(agg, markers=True)
        fig.update_layout(title=title, xaxis_title="Gameweek", yaxis_title="Cumulative Points",
                          plot_bgcolor=COLOUR_TRANSPARENT, paper_bgcolor=COLOUR_TRANSPARENT,
                          font=dict(color=COLOUR_WHITE))
        return html.Div([dcc.Graph(figure=fig)], className="graph-container")
    return build_ranked_colour_table(agg.sort_index(ascending=False).rank(1, ascending=False, method="min"))


@app.callback(
    Output("performance-spread-content", "children"),
    Input("performance-spread-view-toggle", "value"),
    Input("data-reload-trigger", "data"),
)
def update_performance_spread_graph(view_mode, _reload_trigger):
    agg, _ = aggregate_points_by_filter(_merged_df, "all", "total_points")
    if view_mode == "mean":
        pts_mean = agg.mean(0).to_frame()
        pts_std = agg.std(0).to_frame()
        pts_combined = pd.merge(pts_mean, pts_std, left_index=True, right_index=True).reset_index()
        pts_combined.columns = ["User", "Points", "err"]
        fig = px.scatter(pts_combined, x="User", y="Points", error_y="err")
        fig.update_traces(marker_size=10)
        fig.update_xaxes(showgrid=False)
    else:
        fig = px.strip(agg)
    fig.update_layout(
        plot_bgcolor=COLOUR_TRANSPARENT, paper_bgcolor=COLOUR_TRANSPARENT,
        yaxis_title="Points", font=dict(color=COLOUR_WHITE),
    )
    return html.Div([dcc.Graph(figure=fig)], className="graph-container")


# ---------------------------------------------------------------------------
# Squad callbacks — factory pattern
# ---------------------------------------------------------------------------
def _make_squad_metric_callback(output_id: str, filter_id: str, toggle_id: str, metric_col: str):
    @app.callback(
        Output(output_id, "children"),
        Input(filter_id, "value"),
        Input(toggle_id, "value"),
        Input("data-reload-trigger", "data"),
    )
    def _cb(filter_type, view_mode, _reload_trigger):
        agg, title = aggregate_points_by_filter(_merged_df, filter_type, metric_col)
        agg = agg.cumsum()
        if view_mode == "chart":
            fig = px.line(agg, markers=True)
            fig.update_layout(
                title=title,
                xaxis_title="Gameweek",
                yaxis_title=metric_col.replace("_", " ").capitalize(),
                plot_bgcolor=COLOUR_TRANSPARENT,
                paper_bgcolor=COLOUR_TRANSPARENT,
                font=dict(color=COLOUR_WHITE),
            )
            return html.Div([dcc.Graph(figure=fig)], className="graph-container")
        return build_ranked_colour_table(agg.sort_index(ascending=False).rank(1, ascending=False, method="min"))

    return _cb


_make_squad_metric_callback("squad-goals-content", "squad-goals-pos-filter", "squad-goals-view-toggle", "goals_scored")
_make_squad_metric_callback("squad-assists-content", "squad-assists-pos-filter", "squad-assists-view-toggle", "assists")
_make_squad_metric_callback("squad-clean-sheets-content", "squad-clean-sheets-pos-filter", "squad-clean-sheets-view-toggle", "clean_sheets")

# ---------------------------------------------------------------------------
# Player Stats callbacks
# ---------------------------------------------------------------------------
@app.callback(
    Output("player-stats-summary-table", "data"),
    Input("player-stats-pos-filter", "value"),
    Input("data-reload-trigger", "data"),
)
def update_player_stats_table(pos_filter, _reload_trigger):
    summary_df = build_player_stats_summary(_merged_df, pos_filter or "all")
    return summary_df.to_dict("records") if not summary_df.empty else []


@app.callback(
    Output("player-stats-gw-container", "children"),
    Input("player-stats-summary-table", "active_cell"),
    State("player-stats-summary-table", "derived_virtual_data"),
)
def update_player_gw_breakdown(active_cell, virtual_data):
    if not active_cell or not virtual_data:
        return html.P(
            "Click on a player row to view their per-gameweek breakdown.",
            style={"color": "#888"},
        )

    row_idx = active_cell["row"]
    if row_idx >= len(virtual_data):
        return html.Div()

    player_name = virtual_data[row_idx].get("Player", "")
    if not player_name or _merged_df.empty:
        return html.P("No data available for this player.", style={"color": "#888"})

    logger.info("Player Stats: showing GW breakdown for %s", player_name)
    gw_df = build_player_gw_breakdown(_merged_df, player_name)
    if gw_df.empty:
        return html.P(f"No gameweek data found for {player_name}.", style={"color": "#888"})

    return html.Div([
        html.H5(
            f"{player_name} — Per Gameweek Breakdown",
            style={"color": COLOUR_WHITE, "marginBottom": "10px"},
        ),
        dash_table.DataTable(
            data=gw_df.to_dict("records"),
            columns=PLAYER_STATS_GW_COLUMNS,
            sort_action="native",
            style_table={"overflowX": "auto"},
            style_header={
                "backgroundColor": COLOUR_BLACK,
                "color": COLOUR_WHITE,
                "fontWeight": "bold",
                "textAlign": "center",
            },
            style_cell={
                "backgroundColor": COLOUR_DARK_GREY,
                "color": COLOUR_WHITE,
                "textAlign": "center",
                "padding": "8px",
            },
            style_data={"border": "1px solid #555"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#404040"},
            ],
        ),
    ])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
