import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from googleapiclient.discovery import build
from config import CHANNELS

# Load .env from project root, regardless of where the script runs from
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("YOUTUBE_API_KEY")
if not API_KEY:
    raise RuntimeError(
        f"YOUTUBE_API_KEY not found. Looked in: {ROOT / '.env'}"
    )

youtube = build("youtube", "v3", developerKey=API_KEY)

# Output directory at project root
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def get_channel_id(handle: str) -> str:
    """Resolve a channel handle (e.g. @ChrisMD) to a channel ID."""
    res = youtube.channels().list(
        part="id",
        forHandle=handle.lstrip("@"),
    ).execute()
    if not res.get("items"):
        raise ValueError(f"No channel found for handle: {handle}")
    return res["items"][0]["id"]


def get_uploads_playlist(channel_id: str) -> str:
    """Every channel has an 'uploads' playlist containing all its videos."""
    res = youtube.channels().list(
        part="contentDetails",
        id=channel_id,
    ).execute()
    return res["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]


def get_all_video_ids(playlist_id: str) -> list[str]:
    """Page through the uploads playlist to collect every video ID."""
    video_ids, token = [], None
    while True:
        res = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=token,
        ).execute()
        video_ids += [item["contentDetails"]["videoId"] for item in res["items"]]
        token = res.get("nextPageToken")
        if not token:
            break
    return video_ids


def get_video_details(video_ids: list[str]) -> list[dict]:
    """Batch-fetch video metadata 50 at a time."""
    rows = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        res = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(batch),
        ).execute()
        for v in res["items"]:
            rows.append({
                "video_id": v["id"],
                "title": v["snippet"]["title"],
                "published_at": v["snippet"]["publishedAt"],
                "duration": v["contentDetails"]["duration"],
                "views": int(v["statistics"].get("viewCount", 0)),
                "likes": int(v["statistics"].get("likeCount", 0)),
                "comments": int(v["statistics"].get("commentCount", 0)),
                "tags": ",".join(v["snippet"].get("tags", [])),
            })
    return rows


def main():
    all_data = []
    for name, handle in CHANNELS.items():
        print(f"Fetching {name}...")
        try:
            channel_id = get_channel_id(handle)
            playlist_id = get_uploads_playlist(channel_id)
            video_ids = get_all_video_ids(playlist_id)
            print(f"  Found {len(video_ids)} videos")
            videos = get_video_details(video_ids)
            for v in videos:
                v["channel"] = name
            all_data += videos
        except Exception as e:
            print(f"  ERROR fetching {name}: {e}")
            continue

    df = pd.DataFrame(all_data)
    output_path = DATA_DIR / "cmd_videos.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} videos to {output_path}")


if __name__ == "__main__":
    main()