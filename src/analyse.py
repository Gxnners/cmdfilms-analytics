import pandas as pd
import isodate
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "cmd_videos.csv")

# Parse dates and durations
df["published_at"] = pd.to_datetime(df["published_at"])
df["duration_sec"] = df["duration"].apply(lambda x: isodate.parse_duration(x).total_seconds())
df["duration_min"] = df["duration_sec"] / 60
df["year"] = df["published_at"].dt.year
df["month"] = df["published_at"].dt.month
df["day_of_week"] = df["published_at"].dt.day_name()
df["hour"] = df["published_at"].dt.hour
df["engagement_rate"] = (df["likes"] + df["comments"]) / df["views"].replace(0, 1)

# Split into long-form vs shorts (YouTube classifies Shorts as <= 60s)
long_form = df[df["duration_sec"] > 60].copy()
shorts = df[df["duration_sec"] <= 60].copy()

print("=" * 60)
print("CMD FILMS ANALYTICS OVERVIEW")
print("=" * 60)

print(f"\nTotal videos: {len(df):,}")
print(f"Long-form: {len(long_form):,} | Shorts: {len(shorts):,}")
print(f"Total views across all videos: {df['views'].sum():,}")

print("\n--- Per-channel summary ---")
print(df.groupby("channel").agg(
    videos=("video_id", "count"),
    total_views=("views", "sum"),
    avg_views=("views", "mean"),
    median_views=("views", "median"),
    avg_engagement=("engagement_rate", "mean"),
).round(2))

print("\n--- Top 10 long-form videos by views ---")
print(long_form.nlargest(10, "views")[["channel", "title", "views", "duration_min"]].to_string(index=False))

print("\n--- Long-form: best day of week to upload ---")
dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
print(long_form.groupby("day_of_week")["views"].agg(["count", "mean", "median"]).reindex(dow_order).round(0))

print("\n--- Long-form: optimal video length buckets ---")
long_form["length_bucket"] = pd.cut(
    long_form["duration_min"],
    bins=[0, 5, 10, 15, 20, 30, 60, 999],
    labels=["<5m","5-10m","10-15m","15-20m","20-30m","30-60m","60m+"],
)
print(long_form.groupby("length_bucket", observed=True)["views"].agg(["count", "mean", "median"]).round(0))

print("\n--- Yearly performance trend (long-form) ---")
print(long_form.groupby("year").agg(
    videos=("video_id", "count"),
    avg_views=("views", "mean"),
    median_views=("views", "median"),
).round(0))