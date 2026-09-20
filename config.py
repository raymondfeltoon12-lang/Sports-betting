import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = DATA_DIR / "nfl_betting.db"
# Tracked in git (unlike the db, which regenerates from nfl_data_py): live
# odds snapshots capture a moment in time and can't be recreated later.
ODDS_SNAPSHOT_CSV = PROCESSED_DIR / "odds_snapshots.csv"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Seasons to pull for historical training data. nflverse schedule data with
# betting lines reliably goes back further, but pbp-derived team stats get
# noisier pre-2010 (rule changes, tracking gaps), so default to a recent
# 11-season window. Override with env vars if you want more/less history.
START_SEASON = int(os.getenv("START_SEASON", "2015"))
END_SEASON = int(os.getenv("END_SEASON", "2026"))
SEASONS = list(range(START_SEASON, END_SEASON + 1))

# Chronological train/test split: seasons before this are training data,
# this season and later are held out for testing/backtesting. Never split
# randomly -- that would leak future-season information into training.
TEST_START_SEASON = int(os.getenv("TEST_START_SEASON", "2023"))

# Backtesting / Kelly sizing
STARTING_BANKROLL = float(os.getenv("STARTING_BANKROLL", "10000"))
KELLY_MULTIPLIER = float(os.getenv("KELLY_MULTIPLIER", "1.0"))  # 0.5 = half-Kelly
MAX_BET_FRACTION = float(os.getenv("MAX_BET_FRACTION", "0.2"))  # cap per-bet stake as a guard rail

# The Odds API (https://the-odds-api.com) - free tier: current/upcoming odds
# only, 500 requests/month. Historical odds endpoint requires a paid plan,
# so this is used to snapshot live lines going forward (opening vs closing
# movement accumulates over time as the pipeline is re-run).
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
ODDS_API_SPORT = "americanfootball_nfl"
ODDS_API_REGIONS = "us"
ODDS_API_MARKETS = "spreads"
