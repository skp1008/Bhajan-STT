import argparse
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000


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


def build_engine(engine_name: str, cfg: ASRConfig):
    engine_name = engine_name.lower().strip()
    if engine_name == "whisper":
        return WhisperEngine(cfg)
    raise ValueError(f"Unknown engine '{engine_name}'. Only 'whisper' is implemented right now.")


def main():
    p = argparse.ArgumentParser(description="Step 1–4: windowed ASR test on a .wav file (faster-whisper).")
    p.add_argument("--audio", default="test_audio.wav", help="Path to input audio (wav recommended).")
    p.add_argument("--engine", default="whisper", choices=["whisper"], help="ASR engine (only whisper for now).")

    p.add_argument("--model-size", default="small", help="Whisper size: tiny/base/small/medium/large-v3 etc.")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--compute-type", default="int8", help="e.g., int8, int8_float16, float16 (cuda), float32")
    p.add_argument("--language", default="hi", help="Language code (hi for Hindi).")
    p.add_argument("--beam-size", type=int, default=1, help="Beam size (1 is fastest).")

    p.add_argument("--window-sec", type=float, default=15.0, help="Context window length in seconds.")
    p.add_argument("--hop-sec", type=float, default=5.0, help="Hop length in seconds.")
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

    cfg = ASRConfig(
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        beam_size=args.beam_size,
    )
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
    print("-" * 80)

    while t <= end_sec + 1e-6:
        win_start = t - w
        win_end = t

        i0 = int(win_start * TARGET_SR)
        i1 = int(win_end * TARGET_SR)
        window_audio = audio[i0:i1]

        text = engine.transcribe_window(window_audio)

        print(f"[{sec_to_ts(win_start)} -> {sec_to_ts(win_end)}] {text}")

        if args.sleep_real_time:
            time.sleep(h)

        t += h


if __name__ == "__main__":
    main()
