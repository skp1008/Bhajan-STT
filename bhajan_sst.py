import argparse
import sys
import time
import io
import os
import csv
import re
import unicodedata
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000

# --- Lyrics normalization helpers ---

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[""'‘'\(\)\[\]\{\},;:!?]+")
_DANDA_RE = re.compile(r"[।॥]")

def normalize_lyrics_line(s: str) -> str:
    """Conservative normalization for lyrics matching."""
    if s is None:
        return ""
    s = s.strip()
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00A0", " ").replace("\u200B", " ")
    s = _WS_RE.sub(" ", s).strip()
    s = _DANDA_RE.sub("", s)
    s = _PUNCT_RE.sub("", s)
    s = _WS_RE.sub(" ", s).strip()
    return s

def load_lyrics_from_csv(path: str, col_index: int = 0) -> Tuple[List[str], List[str]]:
    """
    Reads CSV and extracts first Devanagari line from each row's first column.
    Returns: (lyrics_base, lyrics_normalized)
    """
    lyrics_base: List[str] = []
    lyrics_normalized: List[str] = []
    
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or col_index >= len(row):
                continue
            cell = row[col_index]
            if cell is None:
                continue
            cell = str(cell).strip()
            if not cell:
                continue
            lines = [ln.strip() for ln in cell.splitlines()]
            first = ""
            for ln in lines:
                if ln:
                    first = ln
                    break
            if not first:
                continue
            lyrics_base.append(first)
            lyrics_normalized.append(normalize_lyrics_line(first))
    
    return lyrics_base, lyrics_normalized


def load_audio_mono(path: str) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32, copy=False)
    return audio, sr


def to_target_sr(audio: np.ndarray, sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    if sr == target_sr:
        return audio
    g = np.gcd(sr, target_sr)
    up = target_sr // g
    down = sr // g
    return resample_poly(audio, up, down).astype(np.float32, copy=False)


def sec_to_ts(sec: float) -> str:
    if sec < 0:
        sec = 0
    m = int(sec // 60)
    s = sec - 60 * m
    return f"{m:02d}:{s:06.3f}"


@dataclass
class ASRConfig:
    model_size: str
    device: str
    compute_type: str
    language: str
    beam_size: int


class WhisperEngine:
    def __init__(self, cfg: ASRConfig):
        from faster_whisper import WhisperModel
        self.cfg = cfg
        self.model = WhisperModel(cfg.model_size, device=cfg.device, compute_type=cfg.compute_type)

    def transcribe_window(self, audio_16k: np.ndarray) -> str:
        segments, info = self.model.transcribe(
            audio_16k,
            language=self.cfg.language,
            beam_size=self.cfg.beam_size,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        texts = []
        for seg in segments:
            t = (seg.text or "").strip()
            if t:
                texts.append(t)
        return " ".join(texts).strip()


class WhisperOnlineEngine:
    def __init__(self, cfg: ASRConfig):
        import openai
        self.cfg = cfg
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Set it with: export OPENAI_API_KEY=your_key")
        self.client = openai.OpenAI(api_key=api_key)

    def transcribe_window(self, audio_16k: np.ndarray) -> str:
        # Write audio to memory buffer
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio_16k, TARGET_SR, format='WAV')
        audio_buffer.seek(0)
        
        # Call OpenAI Whisper API (need filename for format detection)
        transcript = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=("audio.wav", audio_buffer, "audio/wav"),
            language=self.cfg.language,
        )
        return transcript.text.strip()


def build_engine(engine_name: str, cfg: ASRConfig):
    engine_name = engine_name.lower().strip()
    if engine_name == "whisper":
        return WhisperEngine(cfg)
    elif engine_name == "whisper-online":
        return WhisperOnlineEngine(cfg)
    raise ValueError(f"Unknown engine '{engine_name}'. Supported: 'whisper', 'whisper-online'.")


def main():
    p = argparse.ArgumentParser(description="Step 1–4: windowed ASR test on a .wav file (faster-whisper).")
    p.add_argument("--audio", default="test_audio.wav", help="Path to input audio (wav recommended).")
    p.add_argument("--lyrics-csv", type=str, default=None, help="Path to CSV file with lyrics (optional).")
    p.add_argument("--engine", default="whisper", choices=["whisper", "whisper-online"], help="ASR engine: 'whisper' (local) or 'whisper-online' (OpenAI API).")

    p.add_argument("--model-size", default="medium", help="Whisper size: tiny/base/small/medium/large-v3 etc.")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--compute-type", default="int8", help="e.g., int8, int8_float16, float16 (cuda), float32")
    p.add_argument("--language", default="hi", help="Language code (hi for Hindi).")
    p.add_argument("--beam-size", type=int, default=1, help="Beam size (1 is fastest).")

    p.add_argument("--window-sec", type=float, default=15.0, help="Context window length in seconds.")
    p.add_argument("--hop-sec", type=float, default=15.0, help="Hop length in seconds.")
    p.add_argument("--start-sec", type=float, default=0.0, help="Start time in seconds.")
    p.add_argument("--end-sec", type=float, default=-1.0, help="End time in seconds (-1 = full file).")
    p.add_argument("--sleep-real-time", action="store_true",
                   help="If set, sleeps hop-sec between windows to mimic live pace (for demo).")

    args = p.parse_args()

    if args.window_sec <= 0 or args.hop_sec <= 0:
        print("window-sec and hop-sec must be > 0", file=sys.stderr)
        sys.exit(2)
    if args.hop_sec > args.window_sec:
        print("hop-sec should usually be <= window-sec", file=sys.stderr)
        sys.exit(2)

    audio, sr = load_audio_mono(args.audio)
    audio = to_target_sr(audio, sr, TARGET_SR)

    total_sec = len(audio) / TARGET_SR
    start_sec = max(0.0, args.start_sec)
    end_sec = total_sec if args.end_sec < 0 else min(total_sec, args.end_sec)

    # Load lyrics if CSV provided
    lyrics_base: Optional[List[str]] = None
    lyrics_normalized: Optional[List[str]] = None
    if args.lyrics_csv:
        lyrics_base, lyrics_normalized = load_lyrics_from_csv(args.lyrics_csv)

    cfg = ASRConfig(
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        beam_size=args.beam_size,
    )
    
    if args.engine == "whisper-online":
        print("Warning: --model-size, --device, --compute-type, and --beam-size are ignored for whisper-online engine.", file=sys.stderr)
    
    engine = build_engine(args.engine, cfg)

    w = args.window_sec
    h = args.hop_sec

    # Sliding window: [t-w, t], starting when t reaches w
    t = start_sec + w
    if t > end_sec:
        print("Audio segment is shorter than window-sec.", file=sys.stderr)
        sys.exit(2)

    print(f"Loaded: {args.audio}")
    print(f"Duration: {total_sec:.2f}s | Processing range: {start_sec:.2f}s to {end_sec:.2f}s")
    print(f"Engine: {args.engine} | model={args.model_size} | device={args.device} | compute={args.compute_type}")
    print(f"Window: {w:.2f}s | Hop: {h:.2f}s")
    
    if lyrics_base:
        print(f"\nLyrics loaded from: {args.lyrics_csv}")
        print(f"Total lines: {len(lyrics_base)}")
    
    print("-" * 80)

    start_time = time.time()
    while t <= end_sec + 1e-6:
        win_start = t - w
        win_end = t

        i0 = int(win_start * TARGET_SR)
        i1 = int(win_end * TARGET_SR)
        window_audio = audio[i0:i1]

        text = engine.transcribe_window(window_audio)

        elapsed_time = time.time() - start_time
        print(f"\n[{sec_to_ts(win_start)} -> {sec_to_ts(win_end)}] [{elapsed_time:.3f}s elapsed]")
        print(f"ASR Output: {text}")
        
        if lyrics_base:
            print(f"Lyrics (raw): {lyrics_base}")
            print(f"Lyrics (normalized): {lyrics_normalized}")

        if args.sleep_real_time:
            time.sleep(h)

        t += h


if __name__ == "__main__":
    main()


### python bhajan_sst.py --audio yuhhhh.wav --engine whisper-online --model-size medium --device cpu --compute-type int8 --language hi --beam-size 1 --window-sec 30.0 --hop-sec 30.0 --start-sec 0.0 --end-sec -1.0 