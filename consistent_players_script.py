import time
import pandas as pd
import numpy as np
from tqdm import tqdm

from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players


# ----------------------------------
# CONFIG
# ----------------------------------
SEASONS = ["2023-24", "2024-25"]   # update automatically each season
MIN_GAMES = 55
MIN_AVG_MINUTES = 24
SLEEP_TIME = 0.6   # seconds between API calls


# ----------------------------------
# GET ACTIVE PLAYERS
# ----------------------------------
all_players = players.get_active_players()
player_df = pd.DataFrame(all_players)


# ----------------------------------
# FETCH GAME LOGS
# ----------------------------------
game_logs = []

for _, row in tqdm(player_df.iterrows(), total=len(player_df)):
    try:
        for season in SEASONS:
            gl = playergamelog.PlayerGameLog(
                player_id=row["id"],
                season=season
            ).get_data_frames()[0]

            if not gl.empty:
                gl["PLAYER_ID"] = row["id"]
                gl["PLAYER_NAME"] = row["full_name"]
                gl["SEASON"] = season
                game_logs.append(gl)

        time.sleep(SLEEP_TIME)

    except Exception as e:
        print(f"Error fetching {row['full_name']}: {e}")
        continue


# ----------------------------------
# COMBINE DATA
# ----------------------------------
df = pd.concat(game_logs, ignore_index=True)

# ----------------------------------
# CLEAN & FORMAT
# ----------------------------------
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
df["MINUTES"] = pd.to_numeric(df["MIN"], errors="coerce")
df["POINTS"] = pd.to_numeric(df["PTS"], errors="coerce")

df = df.dropna(subset=["MINUTES", "POINTS"])
df = df[df["MINUTES"] > 0]


# ----------------------------------
# AGGREGATE PLAYER STATS
# ----------------------------------
player_stats = (
    df
    .groupby(["PLAYER_ID", "PLAYER_NAME"])
    .agg(
        games_played=("POINTS", "count"),
        avg_minutes=("MINUTES", "mean"),
        mean_points=("POINTS", "mean"),
        std_points=("POINTS", "std"),
    )
    .reset_index()
)

# ----------------------------------
# FILTERS
# ----------------------------------
player_stats = player_stats[
    (player_stats["games_played"] >= MIN_GAMES) &
    (player_stats["avg_minutes"] >= MIN_AVG_MINUTES) &
    (player_stats["mean_points"] > 0)
]

# ----------------------------------
# COEFFICIENT OF VARIATION
# ----------------------------------
player_stats["cv_points"] = (
    player_stats["std_points"] / player_stats["mean_points"]
)

# ----------------------------------
# RANK BY CONSISTENCY
# ----------------------------------
player_stats = player_stats.sort_values("cv_points")

# ----------------------------------
# OPTIONAL: CV PERCENTILES
# ----------------------------------
player_stats["cv_percentile"] = (
    player_stats["cv_points"].rank(pct=True)
)

# ----------------------------------
# OUTPUT
# ----------------------------------
cols = [
    "PLAYER_NAME",
    "games_played",
    "avg_minutes",
    "mean_points",
    "std_points",
    "cv_points",
    "cv_percentile",
]

player_stats[cols].to_csv(
    "nba_scoring_consistency_cv.csv",
    index=False
)

print("\nTop 20 Most Consistent Scorers:\n")
print(player_stats[cols].head(20))