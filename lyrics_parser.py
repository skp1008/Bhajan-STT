import csv
import re
import unicodedata
from typing import List, Tuple

# --- Normalization helpers (lightweight, safe for now) ---

_WS_RE = re.compile(r"\s+")
# Remove common punctuation that doesn't help matching
_PUNCT_RE = re.compile(r"[""'‘'\(\)\[\]\{\},;:!?]+")
# Normalize Hindi danda variants to nothing (optional, but usually helpful)
_DANDA_RE = re.compile(r"[।॥]")

def normalize_lyrics_line(s: str) -> str:
    """
    Conservative normalization:
    - Unicode normalize (NFKC) to reduce weird variants
    - strip
    - collapse whitespace
    - remove some punctuation
    - remove danda/॥ to avoid mismatches across sources
    """
    if s is None:
        return ""

    s = s.strip()
    if not s:
        return ""

    # Normalize unicode forms (handles some lookalikes / width variants)
    s = unicodedata.normalize("NFKC", s)

    # Replace NBSP etc with normal spaces, then collapse
    s = s.replace("\u00A0", " ").replace("\u200B", " ")
    s = _WS_RE.sub(" ", s).strip()

    # Light punctuation cleanup
    s = _DANDA_RE.sub("", s)
    s = _PUNCT_RE.sub("", s)

    # Re-collapse in case punctuation removal created doubles
    s = _WS_RE.sub(" ", s).strip()

    return s


def load_lyrics_from_csv(path: str, col_index: int = 0) -> Tuple[List[str], List[str]]:
    """
    Reads a CSV like your screenshot:
      - Column A: Hindi line(s) + blank line + transliteration
      - Column B: meaning (ignored for now)
    We extract ONLY the first line of Column A per row (the Devanagari line).

    Returns:
      lyrics_base: raw first-line strings (trimmed)
      lyrics_normalized: normalized version of those strings
    """
    lyrics_base: List[str] = []
    lyrics_normalized: List[str] = []

    # utf-8-sig handles BOM if the CSV was exported with it
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue

            if col_index >= len(row):
                continue

            cell = row[col_index]
            if cell is None:
                continue

            cell = str(cell).strip()
            if not cell:
                continue

            # Split into lines; take FIRST non-empty line as "Hindi line"
            # (Some files may have leading empty lines)
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


# -----------------------
# Step 5 usage (example)
# -----------------------
if __name__ == "__main__":
    csv_path = "aho_hari.csv"

    lyrics_base, lyrics_normalized = load_lyrics_from_csv(csv_path)

    print("lyrics_base:")
    for i, line in enumerate(lyrics_base, 1):
        print(f"{i:02d}. {line}")

    print("\nlyrics_normalized:")
    for i, line in enumerate(lyrics_normalized, 1):
        print(f"{i:02d}. {line}")

