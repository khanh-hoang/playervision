# PlayerVision

Takes a football video from a fixed camera and makes a movement heatmap for each player on a top-down pitch.

Pipeline: detect players (YOLO11), track them (ByteTrack), join split tracks, map positions to pitch metres, draw one heatmap per player. FastAPI serves it and a React page shows the results.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd frontend && npm install && npm run build && cd ..
```

## Run

```bash
uvicorn api.main:app --port 8000
```

Open http://localhost:8000 and upload a video. Stop with Ctrl+C.

Background:

```bash
nohup uvicorn api.main:app --port 8000 > server.log 2>&1 &
pkill -f "uvicorn api.main"
```

If the server stops while a video is processing, that job is marked failed. Upload the video again.

Docker (not tested): `docker compose up --build -d`, `docker compose down`

## Usage

1. Upload a video.
2. Find your player on the overview image and note the number.
3. Open the heatmap with that number.

Each heatmap shows what share of the video the player was tracked: high is 85% or more, medium 60-85%, low is below 60%.

Command line only:

```bash
python cli.py track video.mp4 --out tracks.csv
python cli.py export video.mp4 tracks.csv --out output
```

Tests: `pytest`

## Folders

- `vision/`: detection and tracking
- `analytics/`: track merging, pitch calibration (`calibration.json`), heatmaps
- `api/`: FastAPI server, SQLite, one background worker
- `frontend/`: React, TypeScript, Vite

## Limitations

- Fixed camera only. One calibration is used for all videos, so positions are less accurate near the image edges.
- Players can still get a new number in crowds. High coverage does not guarantee the right player.
- Referee and goalkeepers are not separated from the other players.
- Distances are estimates and have not been checked against real data.
- CPU only, one video at a time. A 30 second video takes a few minutes.
