"""Empirical historical hit rates for common player-prop thresholds, computed
from play-by-play data for "featured" players at each position (established
volume, not bench/committee players). This is descriptive analysis for
gauging bet floor/safety, not a predictive model, and not calibrated against
actual sportsbook prop odds (this project doesn't collect historical prop
lines -- only the spreads market). Season averages here include the game
being evaluated (not a leakage-safe pre-game filter like the main pipeline's
features), so treat this as "how often does a player of this caliber clear
this bar", not a forecast for one specific upcoming game.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DIR, RAW_DIR

QB_YARD_LINES = [175, 200, 225, 250, 275]
RB_YARD_LINES = [25, 50, 75, 100]
WR_YARD_LINES = [25, 50, 75, 100]

QB_TIER_YPG = 235       # "good"/established starter
RB_TIER_ATT_PG = 12     # "featured" back
WR_TIER_YDS_PG = 45     # "primary" target


def load_pbp() -> pd.DataFrame:
    cols = [
        "game_id", "season", "posteam", "pass_attempt", "pass_touchdown", "passing_yards",
        "passer_player_name", "rusher_player_name", "rushing_yards", "rush_touchdown",
        "receiver_player_name", "receiving_yards", "complete_pass",
    ]
    frames = [pd.read_parquet(f, columns=cols) for f in sorted(RAW_DIR.glob("pbp_*.parquet"))]
    return pd.concat(frames, ignore_index=True)


def qb_hit_rates(pbp: pd.DataFrame) -> list[dict]:
    passing = pbp[pbp["pass_attempt"] == 1].dropna(subset=["passer_player_name"])
    g = passing.groupby(["game_id", "season", "posteam", "passer_player_name"]).agg(
        attempts=("pass_attempt", "sum"), yards=("passing_yards", "sum"), td=("pass_touchdown", "sum"),
    ).reset_index()
    primary = g.sort_values("attempts", ascending=False).drop_duplicates(["game_id", "posteam"])
    season_avg = primary.groupby(["season", "passer_player_name"])["yards"].mean().reset_index(name="season_ypg")
    primary = primary.merge(season_avg, on=["season", "passer_player_name"])
    good = primary[primary["season_ypg"] >= QB_TIER_YPG]

    rows = []
    n = len(good)
    for line in QB_YARD_LINES:
        rows.append({"position": "QB", "tier": f"season avg >={QB_TIER_YPG} ypg", "prop": f"Over {line} pass yds",
                     "n_games": n, "hit_rate": round(float((good["yards"] > line).mean()), 4)})
    rows.append({"position": "QB", "tier": f"season avg >={QB_TIER_YPG} ypg", "prop": "1+ pass TD",
                 "n_games": n, "hit_rate": round(float((good["td"] >= 1).mean()), 4)})
    rows.append({"position": "QB", "tier": f"season avg >={QB_TIER_YPG} ypg", "prop": "2+ pass TD",
                 "n_games": n, "hit_rate": round(float((good["td"] >= 2).mean()), 4)})
    return rows


def rb_hit_rates(pbp: pd.DataFrame) -> list[dict]:
    rushing = pbp.dropna(subset=["rusher_player_name"])
    rush_g = rushing.groupby(["game_id", "season", "posteam", "rusher_player_name"]).agg(
        attempts=("rusher_player_name", "size"), rush_yards=("rushing_yards", "sum"), rush_td=("rush_touchdown", "sum"),
    ).reset_index().rename(columns={"rusher_player_name": "player"})

    receiving = pbp[pbp["complete_pass"] == 1].dropna(subset=["receiver_player_name"])
    rec_g = receiving.groupby(["game_id", "season", "posteam", "receiver_player_name"]).agg(
        rec_yards=("receiving_yards", "sum"), rec_td=("pass_touchdown", "sum"),
    ).reset_index().rename(columns={"receiver_player_name": "player"})

    merged = rush_g.merge(rec_g, on=["game_id", "season", "posteam", "player"], how="left")
    for c in ["rec_yards", "rec_td"]:
        merged[c] = merged[c].fillna(0)
    merged["any_td"] = (merged["rush_td"] + merged["rec_td"]) >= 1

    primary = merged.sort_values("attempts", ascending=False).drop_duplicates(["game_id", "posteam"])
    season_avg = primary.groupby(["season", "player"])["attempts"].mean().reset_index(name="season_att_pg")
    primary = primary.merge(season_avg, on=["season", "player"])
    featured = primary[primary["season_att_pg"] >= RB_TIER_ATT_PG]

    rows = []
    n = len(featured)
    for line in RB_YARD_LINES:
        rows.append({"position": "RB", "tier": f"season avg >={RB_TIER_ATT_PG} carries/g", "prop": f"Over {line} rush yds",
                     "n_games": n, "hit_rate": round(float((featured["rush_yards"] > line).mean()), 4)})
    rows.append({"position": "RB", "tier": f"season avg >={RB_TIER_ATT_PG} carries/g", "prop": "Anytime TD",
                 "n_games": n, "hit_rate": round(float(featured["any_td"].mean()), 4)})
    return rows


def wr_hit_rates(pbp: pd.DataFrame) -> list[dict]:
    receiving = pbp[pbp["complete_pass"] == 1].dropna(subset=["receiver_player_name"])
    rec_g = receiving.groupby(["game_id", "season", "posteam", "receiver_player_name"]).agg(
        receptions=("receiver_player_name", "size"), rec_yards=("receiving_yards", "sum"), rec_td=("pass_touchdown", "sum"),
    ).reset_index().rename(columns={"receiver_player_name": "player"})

    rushing = pbp.dropna(subset=["rusher_player_name"])
    rush_g = rushing.groupby(["game_id", "season", "posteam", "rusher_player_name"]).agg(
        rush_td=("rush_touchdown", "sum"),
    ).reset_index().rename(columns={"rusher_player_name": "player"})

    merged = rec_g.merge(rush_g, on=["game_id", "season", "posteam", "player"], how="left")
    merged["rush_td"] = merged["rush_td"].fillna(0)
    merged["any_td"] = (merged["rec_td"] + merged["rush_td"]) >= 1

    primary = merged.sort_values("rec_yards", ascending=False).groupby(["game_id", "posteam"]).head(1)

    season_avg = primary.groupby(["season", "player"])["rec_yards"].mean().reset_index(name="season_ypg")
    primary = primary.merge(season_avg, on=["season", "player"])
    featured = primary[primary["season_ypg"] >= WR_TIER_YDS_PG]

    rows = []
    n = len(featured)
    for line in WR_YARD_LINES:
        rows.append({"position": "WR/TE", "tier": f"season avg >={WR_TIER_YDS_PG} rec yds/g", "prop": f"Over {line} rec yds",
                     "n_games": n, "hit_rate": round(float((featured["rec_yards"] > line).mean()), 4)})
    rows.append({"position": "WR/TE", "tier": f"season avg >={WR_TIER_YDS_PG} rec yds/g", "prop": "3+ receptions",
                 "n_games": n, "hit_rate": round(float((featured["receptions"] >= 3).mean()), 4)})
    rows.append({"position": "WR/TE", "tier": f"season avg >={WR_TIER_YDS_PG} rec yds/g", "prop": "5+ receptions",
                 "n_games": n, "hit_rate": round(float((featured["receptions"] >= 5).mean()), 4)})
    rows.append({"position": "WR/TE", "tier": f"season avg >={WR_TIER_YDS_PG} rec yds/g", "prop": "Anytime TD",
                 "n_games": n, "hit_rate": round(float(featured["any_td"].mean()), 4)})
    return rows


def main():
    pbp = load_pbp()
    rows = qb_hit_rates(pbp) + rb_hit_rates(pbp) + wr_hit_rates(pbp)
    out = pd.DataFrame(rows).sort_values(["position", "hit_rate"], ascending=[True, False])
    out_path = PROCESSED_DIR / "prop_hit_rates.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
