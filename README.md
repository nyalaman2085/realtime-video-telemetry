# Real-Time Video Telemetry

A Python computer-vision project that detects and tracks people in webcam/video input, annotates frames, and exports occupancy and dwell-time CSV logs.

This repository is an engineering demonstration, not a validated production surveillance system. Detection quality depends on camera placement, lighting, model behavior, and the configured filters.

## What it does

- Uses Ultralytics YOLO11 (\`yolo11n.pt\`) and its tracking interface.
- Filters detections to the person class with a configurable confidence threshold.
- Requires detections to persist across multiple frames before counting them as stable.
- Allows a short missing-frame grace period before closing a track.
- Writes an annotated MP4 video.
- Exports a timestamped occupancy series and per-track dwell-duration records.
- Cleans up capture devices, video writers, CSV handles, and preview windows when processing stops.

## Architecture

\`\`\`text
Webcam or video file
        |
        v
OpenCV frame capture
        |
        v
YOLO person detection + tracking IDs
        |
        v
Confidence / size / area filters
        |
        v
Persistence and missing-frame handling
        |
        +------> Annotated MP4
        +------> occupancy_log.csv
        +------> dwell_time_log.csv
\`\`\`

## Requirements

- Python 3.10 or newer recommended
- A webcam for live input, or a readable video file
- A desktop environment that supports OpenCV preview windows for interactive live use

Install dependencies:

\`\`\`bash
git clone https://github.com/nyalaman2085/realtime-video-telemetry.git
cd realtime-video-telemetry
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
\`\`\`

The Ultralytics model file is downloaded automatically when it is not present and the environment has network access.

## Run it

### Webcam

\`\`\`bash
python detector.py
\`\`\`

Press \`q\` in the OpenCV preview window to stop. The script writes \`webcam_output.mp4\`, \`occupancy_log.csv\`, and \`dwell_time_log.csv\` in the current working directory.

### Video file

Import and call the function from Python:

\`\`\`python
from detector import run_detector

run_detector(
    source_path="input.mp4",
    output_path="results/annotated.mp4",
)
\`\`\`

The output folder is created if it does not exist. The CSV files are written beside the output video and overwritten on each run; copy or rename them if you need to preserve a previous session.

## Output files

### \`occupancy_log.csv\`

One row per processed frame:

- \`timestamp\`: local timezone-aware timestamp with millisecond precision
- \`stable_person_count\`: number of currently visible tracks that passed the persistence check and filters

This is a per-frame count, not a guaranteed real-world occupancy measurement. It can undercount people who are small, occluded, outside the frame, or rejected by filters.

### \`dwell_time_log.csv\`

One row per stable track when it disappears for longer than the configured grace period or when processing stops:

- \`track_id\`: tracker-provided identifier for this run
- \`entry_timestamp\`: timestamp when the track first passed the persistence threshold
- \`exit_timestamp\`: timestamp when the track was closed
- \`dwell_duration_seconds\`: elapsed processing time from stable-track entry to closure

Track IDs are not persistent identities and may change after occlusion or tracker resets. Dwell duration is an estimate based on the processing clock, not identity recognition.

## Configuration

Edit the constants near the top of \`detector.py\`:

| Setting | Default | Effect |
|---|---:|---|
| \`CONFIDENCE_THRESHOLD\` | \`0.65\` | Minimum detector confidence |
| \`PERSISTENCE_FRAMES\` | \`3\` | Frames required before a track is counted |
| \`MAX_MISSING_FRAMES\` | \`8\` | Missed frames tolerated before a track is closed |
| \`MIN_BOX_WIDTH\` | \`80\` | Minimum bounding-box width in pixels |
| \`MIN_BOX_HEIGHT\` | \`120\` | Minimum bounding-box height in pixels |
| \`MIN_BOX_AREA_RATIO\` | \`0.03\` | Minimum bounding-box fraction of frame area |

The size and area filters intentionally favor larger, closer detections and may reject distant people. Tune them against representative videos instead of treating the defaults as universally correct.

## Validation plan

Before making accuracy or performance claims, evaluate on representative clips with varied lighting, distance, occlusion, and crowd density. Record at least:

- Precision and recall against manually labeled frames
- Missed-person and false-positive examples
- Track ID switches and short occlusion behavior
- Processing FPS and end-to-end latency
- Whether CSV counts match manually checked sample frames

No benchmark values are claimed by this repository until such an evaluation is recorded.

## Known limitations

- Uses a general-purpose pretrained detector; no custom dataset training is included.
- Filters can trade recall for fewer noisy detections.
- Tracking IDs are temporary and not a person's identity.
- The interactive preview requires a graphical desktop; headless servers need a no-preview mode before deployment.
- CSV output is overwritten per run.
- This project is a portfolio/learning demonstration, not a production-ready people-counting or safety system.

## Project structure

\`\`\`text
realtime-video-telemetry/
├── detector.py
├── requirements.txt
├── README.md
├── occupancy_log.csv       # generated when the detector runs
├── dwell_time_log.csv      # generated when the detector runs
└── webcam_output.mp4       # generated when the detector runs
\`\`\`

## Technical summary

Python · OpenCV · Ultralytics YOLO11 · object tracking · CSV telemetry · video processing · resource cleanup

## License

Add a license file before inviting reuse or contributions. Check the model and dependency licenses separately before redistributing their artifacts.
