import asyncio
import os
import sys
import tempfile
import shutil
import subprocess
from pathlib import Path
from threading import Timer

# Check and install dependencies BEFORE importing them
def install_dependencies():
    """Install Python dependencies from requirements.txt"""
    requirements_path = Path(__file__).parent / "requirements.txt"
    if not requirements_path.exists():
        print("Warning: requirements.txt not found")
        return False
    
    try:
        # Try to import critical packages
        import fastapi
        import uvicorn
        return True
    except ImportError:
        print("Installing Python dependencies...")
        try:
            # Read requirements
            with open(requirements_path, "r") as f:
                required_packages = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
            
            subprocess.run(
                [sys.executable, "-m", "pip", "install"] + required_packages,
                check=True
            )
            print("✓ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error installing dependencies: {e}")
            print("Please manually run: pip install -r requirements.txt")
            return False
        except Exception as e:
            print(f"Error checking dependencies: {e}")
            return False

# Install dependencies first
if not install_dependencies():
    print("Failed to install dependencies. Exiting.")
    sys.exit(1)

# Now import the packages (they should be installed)
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import webbrowser

# Import from bhajan_sst.py
sys.path.insert(0, str(Path(__file__).parent))
from bhajan_sst import (
    load_audio_mono, to_target_sr, TARGET_SR,
    load_lyrics_from_csv, extract_anchor_tokens,
    normalize_lyrics_line, score_line_match, decide_line_switch,
    build_engine, ASRConfig, sec_to_ts
)

app = FastAPI()

# Load secrets file if it exists
def load_secrets():
    """Load API key from .secrets file"""
    secrets_path = Path(__file__).parent / ".secrets"
    if secrets_path.exists():
        with open(secrets_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if key == "OPENAI_API_KEY" and value:
                        os.environ["OPENAI_API_KEY"] = value
                        print(f"✓ Loaded API key from .secrets file")
                        return
    # If no .secrets file or no API key, check environment variable
    if os.getenv("OPENAI_API_KEY"):
        print(f"✓ Using API key from environment variable")
    else:
        print("ℹ No API key found (will use local whisper engine)")

# Note: load_secrets() will be called in main()

# Store uploaded files temporarily
UPLOAD_DIR = Path(tempfile.gettempdir()) / "bhajan_stt_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Serve frontend static files
frontend_build = Path(__file__).parent / "frontend" / "dist"
frontend_dir = Path(__file__).parent / "frontend"

# Mount assets directory if it exists (will be created after build)
assets_dir = frontend_build / "assets"
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the React app - dynamically check if built"""
    index_path = frontend_build / "index.html"
    if index_path.exists():
        return index_path.read_text()
    # Frontend not built yet
    return """
    <html>
    <head><title>AI Bhajan Lyrics Listener</title></head>
    <body style="font-family: Arial; padding: 40px; text-align: center; background: #1e1e2e; color: #e0e0e0;">
    <h1>Frontend not built</h1>
    <p>Building frontend... Please wait a moment and refresh.</p>
    <p>If this persists, please run the following commands:</p>
    <pre style="background: #2a2a3e; padding: 20px; display: inline-block; border-radius: 5px; color: #b0b0b0;">
cd frontend
npm install
npm run build
cd ..
python server.py
    </pre>
    </body>
    </html>
    """

@app.get("/{path:path}")
async def serve_static(path: str):
    """Serve other static files"""
    from fastapi.responses import FileResponse
    file_path = frontend_build / path
    if file_path.exists() and file_path.is_file():
        return FileResponse(str(file_path))
    # Fallback to index.html for client-side routing
    index_path = frontend_build / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    # If frontend not built, return the same message as root
    return HTMLResponse(content="""
    <html>
    <head><title>AI Bhajan Lyrics Listener</title></head>
    <body style="font-family: Arial; padding: 40px; text-align: center; background: #1e1e2e; color: #e0e0e0;">
    <h1>Frontend not built</h1>
    <p>Building frontend... Please wait a moment and refresh.</p>
    </body>
    </html>
    """)


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
        start_sec = 0.0
        end_sec = -1.0
        
        # Initialize engine with defaults
        engine_name = "whisper-online"
        cfg = ASRConfig(
            model_size="medium",
            device="cpu",
            compute_type="int8",
            language="hi",
            beam_size=1
        )
        
        # Fallback to local whisper if no API key
        if not os.getenv("OPENAI_API_KEY"):
            engine_name = "whisper"
            print("Warning: No API key found, using local whisper engine instead of whisper-online")
        
        print(f"Initializing {engine_name} engine...")
        engine = build_engine(engine_name, cfg)
        print(f"Engine initialized. Processing audio ({total_sec:.1f}s total)...")
        
        current_line = 1
        
        # Apply start_sec and end_sec
        start_sec = max(0.0, start_sec)
        end_sec = total_sec if end_sec < 0 else min(total_sec, end_sec)
        total_sec_processed = end_sec - start_sec
        
        # Process windows
        t = start_sec + window_sec
        if t > end_sec:
            await websocket.send_json({"type": "error", "message": "Audio segment is shorter than window-sec."})
            return
        
        window_count = 0
        total_windows = int((end_sec - start_sec - window_sec) / hop_sec) + 1 if total_sec_processed >= window_sec else 1
        print(f"Will process {total_windows} windows (25s each, 5s hop)")
        
        while t <= end_sec + 1e-6:
            win_start = t - window_sec
            win_end = t
            
            i0 = int(win_start * TARGET_SR)
            i1 = int(win_end * TARGET_SR)
            window_audio = audio[i0:i1]
            
            # Transcribe
            print(f"Processing window {window_count + 1}/{total_windows} ({win_start:.1f}s - {win_end:.1f}s)...")
            try:
                asr_text = engine.transcribe_window(window_audio)
                print(f"  Transcribed: {asr_text[:50]}...")
            except Exception as e:
                print(f"  ERROR during transcription: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Transcription error: {str(e)}"
                })
                return
            
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
        print("WebSocket disconnected by client")
        pass
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"ERROR in websocket_process: {e}")
        print(error_trace)
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass
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


def check_and_install_remaining_dependencies():
    """Check for any remaining missing dependencies and install them"""
    requirements_path = Path(__file__).parent / "requirements.txt"
    if not requirements_path.exists():
        return
    
    try:
        # Try to import all required packages
        with open(requirements_path, "r") as f:
            required_packages = []
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    pkg_name = line.split(">=")[0].split("==")[0].split("[")[0].strip()
                    try:
                        __import__(pkg_name.replace("-", "_"))
                    except ImportError:
                        required_packages.append(line)
        
        if required_packages:
            print(f"Installing {len(required_packages)} additional packages...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install"] + required_packages,
                check=True
            )
            print("✓ Additional dependencies installed")
    except Exception as e:
        print(f"Note: Some dependencies may be missing: {e}")


def find_npm():
    """Try to find npm executable"""
    # Try common locations
    possible_paths = [
        "npm",  # In PATH
        "/usr/local/bin/npm",
        "/opt/homebrew/bin/npm",
        shutil.which("npm"),  # Use shutil.which
    ]
    for npm_path in possible_paths:
        if npm_path and (Path(npm_path).exists() or shutil.which(npm_path)):
            return npm_path
    return None

def build_frontend():
    """Build frontend if needed"""
    if not frontend_build.exists() and frontend_dir.exists():
        print("Frontend not built. Building...")
        
        # Find npm
        npm_path = find_npm()
        if not npm_path:
            print("❌ npm not found!")
            print("Please install Node.js from https://nodejs.org/")
            print("After installing, run: cd frontend && npm install && npm run build")
            return False
        
        try:
            original_dir = os.getcwd()
            os.chdir(frontend_dir)
            
            if not (frontend_dir / "node_modules").exists():
                print("Installing npm dependencies...")
                result = subprocess.run([npm_path, "install"], check=True, capture_output=True, text=True)
                if result.stdout:
                    # Only show important output
                    lines = result.stdout.split('\n')
                    for line in lines[-10:]:  # Last 10 lines
                        if line.strip():
                            print(line)
            
            print("Building frontend...")
            result = subprocess.run([npm_path, "run", "build"], check=True, capture_output=True, text=True)
            if result.stdout:
                # Only show important output
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'built in' in line.lower() or 'dist' in line.lower() or 'error' in line.lower():
                        print(line)
                    elif len(lines) < 20:  # If output is short, show all
                        print(line)
            
            os.chdir(original_dir)
            
            # Verify build succeeded
            if frontend_build.exists() and (frontend_build / "index.html").exists():
                print("✓ Frontend built successfully!")
                return True
            else:
                print("⚠ Build completed but index.html not found")
                return False
        except subprocess.CalledProcessError as e:
            print(f"❌ Error building frontend:")
            if e.stderr:
                print(e.stderr)
            if e.stdout:
                # Show last few lines of output
                lines = e.stdout.split('\n')
                for line in lines[-5:]:
                    if line.strip():
                        print(line)
            print("\nPlease manually run: cd frontend && npm install && npm run build")
            try:
                os.chdir(Path(__file__).parent)
            except:
                pass
            return False
        except Exception as e:
            print(f"❌ Unexpected error building frontend: {e}")
            try:
                os.chdir(Path(__file__).parent)
            except:
                pass
            return False
    elif frontend_build.exists():
        print("✓ Frontend already built")
        return True
    else:
        print("⚠ Frontend directory not found")
        return False


if __name__ == "__main__":
    print("="*50)
    print("AI Bhajan Lyrics Listener - Server Setup")
    print("="*50 + "\n")
    
    # Check for any remaining dependencies (dependencies already checked at import time)
    check_and_install_remaining_dependencies()
    print()
    
    # Load secrets
    load_secrets()
    print()
    
    # Build frontend if needed (MUST happen before server starts)
    frontend_ready = build_frontend()
    if not frontend_ready:
        print("\n⚠ Frontend build failed. Server will start but frontend may not work.")
        print("Please manually build the frontend: cd frontend && npm install && npm run build\n")
    else:
        # Remount assets if they were just created
        if assets_dir.exists() and not any(r.path == "/assets" for r in app.routes):
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
    print()
    
    # Open browser after 2 seconds (give build time if it just finished)
    Timer(2.0, open_browser).start()
    
    print("="*50)
    print("Starting server at http://localhost:8000")
    print("Browser will open automatically...")
    print("="*50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)

