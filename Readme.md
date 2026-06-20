# Audio Fingerprinting — EE200 Project Demo

Shazam-style audio fingerprinting app built for the EE200 (Signals, Systems & Networks) course project.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Build the fingerprint database** (run once — takes ~2 min):
   ```bash
   python build_db.py
   ```

3. **Generate sample query clips** (for the Identify tab):
   ```bash
   python generate_samples.py
   ```

4. **Launch the app:**
   ```bash
   streamlit run app.py
   ```

## How It Works

1. **Spectrogram** — each song is turned into a time-frequency image using the STFT.
2. **Constellation** — local maxima (peaks) in the spectrogram are extracted as landmark points.
3. **Hashing** — nearby peaks are paired into compact hashes `(f1, f2, Δt)` for efficient lookup.
4. **Matching** — query hashes are looked up in the pre-built database; matching hashes vote for a time offset, and a true match produces a sharp spike in the offset histogram.

## App Tabs

| Tab | What it does |
|-----|-------------|
| **Library** | Lists all indexed songs with duration and fingerprint stats |
| **Identify** | Upload a clip (or pick a sample) → see the full pipeline: spectrogram, constellation, offset histogram, matched song |
| **Batch** | Upload multiple clips → get a `results.csv` with `filename, prediction` columns |

## File Structure

| File | Purpose |
|------|---------|
| `fingerprint.py` | Core fingerprinting engine |
| `build_db.py` | Indexes all songs into pickle databases |
| `generate_samples.py` | Creates sample query clips |
| `app.py` | Streamlit web app |
| `Q3_data/` | Song library (50 .mp3 files) |
| `samples/` | Sample query clips |
| `song_database.pkl` | Pre-built hash database |
| `song_metadata.pkl` | Song metadata (durations, peaks) |


