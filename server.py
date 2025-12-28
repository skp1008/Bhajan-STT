import asyncio
import os
import sys
import tempfile
import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import webbrowser
from threading import Timer

# Import from bhajan_sst.py
sys.path.insert(0, str(Path(__file__).parent))
from bhajan_sst import (
    load_audio_mono, to_target_sr, TARGET_SR,
    load_lyrics_from_csv, extract_anchor_tokens,
    normalize_lyrics_line, score_line_match, decide_line_switch,
    build_engine, ASRConfig, sec_to_ts
)

app = FastAPI()

# Store uploaded files temporarily
UPLOAD_DIR = Path(tempfile.gettempdir()) / "bhajan_stt_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Serve frontend static files
frontend_build = Path(__file__).parent / "frontend" / "dist"
if frontend_build.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_build)), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the React app"""
    index_path = frontend_build / "index.html"
    if index_path.exists():
        return index_path.read_text()
    return "<html><body><h1>Frontend not built. Run 'npm run build' in frontend directory.</h1></body></html>"


@app.post("/api/upload")
async def upload_files(
    csv_file: UploadFile = File(...),
    audio_file: UploadFile = File(...)
):
    """Handle file uploads"""
    csv_filename = f"csv_{csv_file.filename}"
    audio_filename = f"audio_{audio_file.filename}"
    
    csv_path = UPLOAD_DIR / csv_filename
    audio_path = UPLOAD_DIR / audio_filename
    
    with open(csv_path, "wb") as f:
        shutil.copyfileobj(csv_file.file, f)
    
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio_file.file, f)
    
    return {
        "csv_filename": csv_filename,
        "audio_filename": audio_filename
    }


@app.websocket("/ws/process")
async def websocket_process(websocket: WebSocket):
    await websocket.accept()
    
    try:
        # Get parameters
        params = dict(websocket.query_params)
        csv_filename = params.get("csv_file")
        audio_filename = params.get("audio_file")
        
        if not csv_filename or not audio_filename:
            await websocket.send_json({"type": "error", "message": "Missing file parameters"})
            return
        
        csv_path = UPLOAD_DIR / csv_filename
        audio_path = UPLOAD_DIR / audio_filename
        
        if not csv_path.exists() or not audio_path.exists():
            await websocket.send_json({"type": "error", "message": "Files not found"})
            return
        
        # Load lyrics
        lyrics_base, lyrics_normalized = load_lyrics_from_csv(str(csv_path))
        line_anchors = [extract_anchor_tokens(line) for line in lyrics_normalized]
        
        await websocket.send_json({
            "type": "lyrics_loaded",
            "lyrics": lyrics_base
        })
        
        # Load audio
        audio, sr = load_audio_mono(str(audio_path))
        audio = to_target_sr(audio, sr, TARGET_SR)
        
        total_sec = len(audio) / TARGET_SR
        window_sec = 25.0
        hop_sec = 5.0
        min_score = 2.0
        margin = 1.0
        
        # Initialize engine (use whisper-online by default, can be made configurable)
        cfg = ASRConfig(
            model_size="medium",
            device="cpu",
            compute_type="int8",
            language="hi",
            beam_size=1
        )
        
        # Check for API key to decide engine
        engine_name = "whisper-online" if os.getenv("OPENAI_API_KEY") else "whisper"
        engine = build_engine(engine_name, cfg)
        
        current_line = 1
        
        # Process windows
        t = window_sec
        window_count = 0
        total_windows = int((total_sec - window_sec) / hop_sec) + 1
        
        while t <= total_sec + 1e-6:
            win_start = t - window_sec
            win_end = t
            
            i0 = int(win_start * TARGET_SR)
            i1 = int(win_end * TARGET_SR)
            window_audio = audio[i0:i1]
            
            # Transcribe
            asr_text = engine.transcribe_window(window_audio)
            
            # Line matching
            asr_normalized = normalize_lyrics_line(asr_text)
            asr_tokens = extract_anchor_tokens(asr_normalized)
            
            scores = []
            coverages = []
            for anchors in line_anchors:
                score, coverage = score_line_match(asr_tokens, anchors)
                scores.append(score)
                coverages.append(coverage)
            
            should_switch, new_line = decide_line_switch(
                current_line, scores, coverages, min_score, margin
            )
            
            action = "switch" if should_switch else "stay"
            if should_switch:
                current_line = new_line
            
            timestamp = f"[{sec_to_ts(win_start)} -> {sec_to_ts(win_end)}]"
            
            await websocket.send_json({
                "type": "asr_output",
                "timestamp": timestamp,
                "asr_text": asr_text,
                "action": action,
                "line_number": current_line,
                "lyrics_line": lyrics_base[current_line - 1] if current_line <= len(lyrics_base) else ""
            })
            
            window_count += 1
            progress = int((window_count / total_windows) * 100)
            await websocket.send_json({"type": "progress", "progress": progress})
            
            t += hop_sec
        
        await websocket.send_json({"type": "complete"})
        
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
    finally:
        # Cleanup uploaded files
        try:
            if csv_filename:
                csv_path = UPLOAD_DIR / csv_filename
                if csv_path.exists():
                    csv_path.unlink()
            if audio_filename:
                audio_path = UPLOAD_DIR / audio_filename
                if audio_path.exists():
                    audio_path.unlink()
        except:
            pass


def open_browser():
    """Open browser after a short delay"""
    webbrowser.open('http://localhost:8000')


if __name__ == "__main__":
    # Build frontend if needed
    frontend_dir = Path(__file__).parent / "frontend"
    if not frontend_build.exists() and frontend_dir.exists():
        print("Frontend not built. Building...")
        import subprocess
        os.chdir(frontend_dir)
        subprocess.run(["npm", "install"], check=True)
        subprocess.run(["npm", "run", "build"], check=True)
    
    # Open browser after 1.5 seconds
    Timer(1.5, open_browser).start()
    
    print("Starting server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

