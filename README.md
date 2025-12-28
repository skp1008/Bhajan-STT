# AI Bhajan Lyrics Listener

Real-time Devanagari lyrics synchronization using ASR (Automatic Speech Recognition) for Hindu temple kirtans.

## Features

- Real-time speech-to-text using Whisper (local or online)
- Automatic line detection and switching
- Beautiful React-based web interface
- Devanagari script support
- CSV-based lyrics management

## Setup

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies

```bash
cd frontend
npm install
npm run build
cd ..
```

### 3. Set Environment Variable (Optional - for online Whisper)

If you want to use the online Whisper API:

```bash
export OPENAI_API_KEY=your_api_key_here
```

## Usage

### Run the Web Application

```bash
python server.py
```

This will:
1. Start the FastAPI server on http://localhost:8000
2. Automatically open your browser
3. Serve the React frontend

### In the Web Interface

1. Upload a CSV file with lyrics (first column should contain Devanagari text)
2. Upload an audio file (WAV recommended)
3. Click "Start Processing"
4. Watch real-time ASR output and synchronized lyrics

### Run CLI Version (Original)

```bash
python bhajan_sst.py --audio your_audio.wav --lyrics-csv your_lyrics.csv --engine whisper
```

## Project Structure

```
.
├── bhajan_sst.py      # Core ASR and line matching logic
├── server.py          # FastAPI web server
├── frontend/          # React frontend
│   ├── src/
│   │   ├── App.jsx    # Main React component
│   │   └── App.css    # Styles
│   └── package.json
└── requirements.txt   # Python dependencies
```

## Technologies

- **Backend**: Python, FastAPI, WebSocket
- **Frontend**: React, Vite
- **ASR**: faster-whisper (local) or OpenAI Whisper API (online)
- **Audio Processing**: soundfile, scipy

