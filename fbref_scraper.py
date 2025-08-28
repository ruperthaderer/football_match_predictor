# fbref_scraper.py
# Scraped FBref Spieler-Tabellen (summary + keeper) je Match
# und schreibt ALLE Ligen/Jahre in EINE CSV (append mit Resume).
#
# Install:
#   pip install playwright bs4 pandas lxml tqdm
#   python -m playwright install chromium
#
# Run:
#   python fbref_scraper.py

import os
import re
import csv
import time
import random
import asyncio
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, Comment
from tqdm import tqdm
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

# --------------------------------------------
# Konfiguration
# --------------------------------------------
BASE = "https://fbref.com"

# Liga-Slugs und comp_id (aus deiner Struktur)
LEAGUES = {
    "Premier-League": 9,
    "La-Liga": 12,
    "Bundesliga": 20,
    "Serie-A": 11,
    "Ligue-1": 13,
}

YEARS = list(range(2000, 2025 + 1))  # Saison 2000/01 .. 2024/25
OUTFILE = "players_fbref_2000_2025.csv"
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Für Tests kannst du limitieren:
MAX_MATCHES_PER_SEASON = None #. 10, um erstmal zu testen; sonst None

# --------------------------------------------
# Helper
# --------------------------------------------
def season_str(y: int) -> str:
    return f"{y}-{y+1}"

def schedule_url(league_slug: str, comp_id: int, s: str) -> str:
    # Beispiel: /en/comps/9/2024-2025/schedule/2024-2025-Premier-League-Scores-and-Fixtures
    return f"{BASE}/en/comps/{comp_id}/{s}/schedule/{s}-{league_slug}-Scores-and-Fixtures"

def remove_html_comments(html: str) -> str:
    """FBref bettet viele Tabellen in <!-- ... --> ein. Kommentare entfernen, Inhalt behalten."""
    soup = BeautifulSoup(html, "lxml")
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        c.replace_with(c)
    return str(soup)

def checkpoint_path(league_slug: str, season_end: int) -> Path:
    return CHECKPOINT_DIR / f"done_{league_slug}_{season_end}.flag"

def season_done(league_slug: str, season_end: int) -> bool:
    return checkpoint_path(league_slug, season_end).exists()

def mark_season_done(league_slug: str, season_end: int) -> None:
    checkpoint_path(league_slug, season_end).write_text("ok", encoding="utf-8")

def append_csv(df: pd.DataFrame, outfile: str) -> None:
    file_exists = os.path.exists(outfile)
    df.to_csv(outfile, mode="a", index=False, header=not file_exists, quoting=csv.QUOTE_MINIMAL)

# --------------------------------------------
# Parsing
# --------------------------------------------
def parse_match_players_from_html(html: str, league_slug: str, sstr: str, match_url: str) -> pd.DataFrame:
    soup = BeautifulSoup(remove_html_comments(html), "lxml")
    title_el = soup.select_one("h1")
    title = title_el.get_text(strip=True) if title_el else None

    frames = []
    # Tabellen wie 'stats_manchester-united_summary' und 'stats_manchester-united_keeper'
    for tbl in soup.find_all("table", id=re.compile(r"^stats_.*_(summary|keeper)$")):
        tbl_id = tbl.get("id", "")
        m = re.match(r"^stats_(.+?)_(summary|keeper)$", tbl_id)
        team_hint = m.group(1) if m else None
        stat_type = m.group(2) if m else None

        try:
            df = pd.read_html(str(tbl))[0]
        except ValueError:
            continue

        # Multiindex-Header abflachen
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(c[-1]) for c in df.columns]
        else:
            df.columns = [str(c) for c in df.columns]

        # "Player"-Spalte sicherstellen
        if "Player" not in df.columns:
            for c in df.columns:
                if "Player" in c or "Unnamed" in c.lower():
                    df = df.rename(columns={c: "Player"})
                    break

        df["match_url"] = match_url
        df["league"] = league_slug
        df["season"] = sstr
        df["table_type"] = stat_type
        df["team_hint"] = team_hint
        df["match_title"] = title
        frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# --------------------------------------------
# Playwright-basierte Loader
# --------------------------------------------
async def list_match_urls(page, sched_url: str) -> list[str]:
    # Schedule-Seite laden
    await page.goto(sched_url, wait_until="domcontentloaded", timeout=120_000)
    # kleine menschliche Pause
    await page.wait_for_timeout(900 + int(random.random() * 700))
    anchors = await page.query_selector_all('td[data-stat="score"] a[href^="/en/matches/"]')
    hrefs = []
    for a in anchors:
        href = await a.get_attribute("href")
        if href:
            hrefs.append(BASE + href)
    # eindeutige Reihenfolge beibehalten
    seen, uniq = set(), []
    for h in hrefs:
        if h not in seen:
            seen.add(h)
            uniq.append(h)
    return uniq

async def scrape_season_playwright(league_slug: str, comp_id: int, season_end: int) -> pd.DataFrame:
    sstr = season_str(season_end - 1)
    sched = schedule_url(league_slug, comp_id, sstr)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ))
        page = await ctx.new_page()

        # Warmup
        await page.goto(f"{BASE}/", wait_until="domcontentloaded", timeout=60_000)
        await page.wait_for_timeout(800 + int(random.random() * 600))

        # Fixture-Links einsammeln
        urls = await list_match_urls(page, sched)
        if MAX_MATCHES_PER_SEASON:
            urls = urls[:MAX_MATCHES_PER_SEASON]

        out = []
        for mu in tqdm(urls, desc=f"{league_slug} {sstr}"):
            try:
                # Matchseite
                await page.goto(mu, wait_until="domcontentloaded", timeout=120_000)
                await page.wait_for_timeout(900 + int(random.random() * 800))
                html = await page.content()
                df = parse_match_players_from_html(html, league_slug, sstr, mu)
                if not df.empty:
                    out.append(df)
            except PlaywrightTimeout as e:
                print(f"  ⚠️ timeout: {mu} ({e})")
            except Exception as e:
                print(f"  ⚠️ skip {mu}: {e}")
            # höfliche Pause zwischen Matches
            await page.wait_for_timeout(1200 + int(random.random() * 1200))

        await browser.close()

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

# --------------------------------------------
# Main: Loop über alle Ligen & Jahre mit Resume
# --------------------------------------------
async def main():
    # Wenn OUTFILE nicht da → erzeugen mit nichts; Header schreibt append_csv beim ersten Append automatisch
    if not os.path.exists(OUTFILE):
        Path(OUTFILE).touch()

    for league_slug, comp_id in LEAGUES.items():
        for season_end in YEARS:
            # Überspringen, wenn es eine fertig-Flag gibt
            if season_done(league_slug, season_end):
                print(f"⏭  {league_slug} {season_end}: already done (checkpoint)")
                continue

            # Scrape
            print(f"▶  {league_slug} {season_end} …")
            try:
                df = await scrape_season_playwright(league_slug, comp_id, season_end)
            except Exception as e:
                print(f"❌  {league_slug} {season_end}: scrape error: {e}")
                # etwas warten und nächste Season versuchen
                time.sleep(10)
                continue

            if df is None or df.empty:
                print(f"⚠️  {league_slug} {season_end}: no rows")
                # trotzdem Flag? Besser NICHT, damit du später neu versuchen kannst.
                time.sleep(8)
                continue

            # Metadaten für Resume/Analyse
            df["league_slug"] = league_slug
            df["season_end"] = season_end

            try:
                append_csv(df, OUTFILE)
                mark_season_done(league_slug, season_end)
                print(f"✅  {league_slug} {season_end}: appended {len(df):,} rows")
            except Exception as e:
                print(f"❌  {league_slug} {season_end}: write/append error: {e}")
                # nicht als done markieren

            # Pause zwischen Seasons (wichtig gegen Rate-Limits)
            time.sleep(12 + random.uniform(0, 6))

    print(f"\n💾 All data continuously written into {OUTFILE}")
    print("✅ Checkpoints under:", CHECKPOINT_DIR.resolve())

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Abgebrochen per KeyboardInterrupt. Resume ist möglich – einfach erneut starten.")
