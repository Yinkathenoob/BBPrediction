import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguegamelog


# -----------------------
# CONFIG
# -----------------------
SEASONS = ["2024-25", "2025-26"]
MIN_GAMES = 55
MIN_AVG_MINUTES = 24


# -----------------------
# FETCH LEAGUE DATA
# -----------------------
all_logs = []

for season in SEASONS:
    print(f"Fetching {season}...")
    
    gamelog = leaguegamelog.LeagueGameLog(
        season=season,
        player_or_team_abbreviation="P"  # P = Player logs
    ).get_data_frames()[0]
    
    gamelog["SEASON"] = season
    all_logs.append(gamelog)

df = pd.concat(all_logs, ignore_index=True)


# -----------------------
# CLEAN DATA
# -----------------------
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
df["MINUTES"] = pd.to_numeric(df["MIN"], errors="coerce")
df["POINTS"] = pd.to_numeric(df["PTS"], errors="coerce")

df = df.dropna(subset=["MINUTES", "POINTS"])
df = df[df["MINUTES"] > 0]


# -----------------------
# AGGREGATE PLAYER STATS
# -----------------------
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

# -----------------------
# APPLY FILTERS
# -----------------------
player_stats = player_stats[
    (player_stats["games_played"] >= MIN_GAMES) &
    (player_stats["avg_minutes"] >= MIN_AVG_MINUTES) &
    (player_stats["mean_points"] > 0)
]


# -----------------------
# CALCULATE CV
# -----------------------
player_stats["cv_points"] = (
    player_stats["std_points"] / player_stats["mean_points"]
)

player_stats = player_stats.sort_values("cv_points")


# -----------------------
# SAVE
# -----------------------
cols = [
    "PLAYER_NAME",
    "games_played",
    "avg_minutes",
    "mean_points",
    "std_points",
    "cv_points",
]

player_stats[cols].to_csv(
    "nba_scoring_consistency_cv.csv",
    index=False
)

print("\nTop 20 Most Consistent Players:\n")
print(player_stats[cols].head(20))