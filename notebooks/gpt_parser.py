#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import os
import re
import time
from pathlib import Path
from typing import List, Tuple
from openai import AzureOpenAI

# ======================
# Config — adjust freely
# ======================
OCR_OUTPUT_DIR = Path("ocr_output")
SYSTEM_MESSAGE_FILE = Path("prompts/system_message.txt")
PARSED_RESULTS_CSV = Path("data/parsed_results.csv")
PROCESSING_LOG_CSV = Path("data/processing_log.csv")

# Processing log fields
LOG_FILENAME_COL = "filename"      # e.g., "image19857.pdf"
LOG_STATUS_COL = "parsed_status"   # values: "pending" | "parsed" | "failed"
STATUS_PENDING = "pending"
STATUS_PARSED = "parsed"
STATUS_FAILED = "failed"

# Batch / run control
MAX_FILES = 10        # None for no limit
MAX_MINUTES = 10      # None for no time limit
MAX_RETRIES = 2       # GPT call retries on failure (per-chunk)

# Output CSV columns (model fields + enrichments)
OUTPUT_FIELDS = [
    "patient_name",
    "patient_id",
    "account_id",
    "claim_id",
    "min_date_of_service",
    "max_date_of_service",
    "letter_date",
    "days_until_due",
    "remit_amount",
    "remit_reason",
    "remit_due_date",
    "parsing_notes",
    "no_records",
    "filename",   # original PDF name from log
    "payor",      # derived from OCR_OUTPUT_DIR/<PAYOR>/...
]

# ======================
# Chunking controls
# ======================
# Try to split on likely patient boundaries first (case-insensitive, ^ = start of line).
PATIENT_SEPARATORS_PATTERN = re.compile(
    r"(?im)^(?:\s*(?:patient(?:\s+name)?|account\s*id|claim\s*id)\b.*)$"
)

# How many detected records (separators) to bundle per chunk when using the regex strategy
SEP_GROUP_SIZE = 30

# If we can't find separators, fall back to size-based chunking:
MAX_CHARS_PER_CHUNK = 12000   # ~3k tokens (rule of thumb: ~4 chars/token)
CHUNK_OVERLAP_CHARS = 1000    # small overlap to reduce boundary issues

# ======================
# Azure OpenAI (hard-coded for now)
# ======================
AZURE_OPENAI_ENDPOINT = "https://my-overpayment-parser.openai.azure.com/"
AZURE_OPENAI_API_KEY = "YOUR_KEY_HERE"
AZURE_OPENAI_API_VERSION = "2024-10-21"
AZURE_OPENAI_DEPLOYMENT = "gpt-4.1-overpayment"
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 4000  # request a high cap per chunk

def make_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )

# ======================
# Helpers
# ======================
def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")

def load_system_message(path: Path) -> str:
    return read_text(path).strip()

def derive_payor(ocr_root: Path, file_path: Path) -> str:
    """First subfolder under OCR_OUTPUT_DIR is treated as payor."""
    try:
        rel = file_path.resolve().relative_to(ocr_root.resolve())
        return rel.parts[0] if len(rel.parts) > 1 else ""
    except Exception:
        return ""

def sanitize_jsonl(raw: str):
    """Keep only lines that look like JSON objects; strip code fences/trailing commas."""
    for line in raw.splitlines():
        s = line.strip().strip("`").strip()
        if not s:
            continue
        if s.endswith(","):
            s = s[:-1]
        if s.startswith("{") and s.endswith("}"):
            yield s

def append_rows(rows, csv_path: Path):
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerows(rows)

def call_gpt(client: AzureOpenAI, system_msg: str, user_text: str) -> str:
    resp = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_text},
        ],
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    return resp.choices[0].message.content or ""

def call_gpt_with_retries(client: AzureOpenAI, system_msg: str, user_text: str) -> str:
    """Linear backoff retries honoring MAX_RETRIES."""
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return call_gpt(client, system_msg, user_text)
        except Exception as e:
            last_err = e
            print(f"[retry {attempt}/{MAX_RETRIES}] GPT call failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)
    raise last_err if last_err else RuntimeError("Unknown GPT error")

# ----------------------
# Chunking implementations
# ----------------------
def find_separator_indices(text: str) -> List[int]:
    """Return starting indices of lines that look like patient boundaries."""
    return [m.start() for m in PATIENT_SEPARATORS_PATTERN.finditer(text)]

def chunks_by_separators(text: str) -> List[Tuple[int, int]]:
    """
    Build chunks using detected patient boundaries.
    Returns list of (start, end) index tuples covering entire text.
    Groups SEP_GROUP_SIZE records per chunk; also bounds by MAX_CHARS_PER_CHUNK.
    """
    n = len(text)
    idxs = find_separator_indices(text)
    if len(idxs) < 2:
        return []  # signal caller to fall back to size-based

    # Always include 0 as a start if the first patient doesn't start at 0
    if idxs[0] != 0:
        idxs = [0] + idxs

    ranges: List[Tuple[int, int]] = []
    i = 0
    while i < len(idxs):
        # Propose a chunk that spans SEP_GROUP_SIZE records
        j = min(i + SEP_GROUP_SIZE, len(idxs))
        start = idxs[i]
        end = idxs[j] if j < len(idxs) else n

        # If the proposed chunk exceeds MAX_CHARS_PER_CHUNK, split it further by size
        if end - start > MAX_CHARS_PER_CHUNK:
            ranges.extend(chunks_by_size(text[start:end], base_offset=start))
        else:
            ranges.append((start, end))
        i = j
    return coalesce_with_overlap(text, ranges, CHUNK_OVERLAP_CHARS)

def chunks_by_size(text: str, base_offset: int = 0) -> List[Tuple[int, int]]:
    """Fallback: fixed-size windows with overlap."""
    n = len(text)
    ranges: List[Tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(start + MAX_CHARS_PER_CHUNK, n)
        ranges.append((base_offset + start, base_offset + end))
        if end >= n:
            break
        start = end - CHUNK_OVERLAP_CHARS if end - CHUNK_OVERLAP_CHARS > start else end
    return ranges

def coalesce_with_overlap(text: str, ranges: List[Tuple[int, int]], overlap: int) -> List[Tuple[int, int]]:
    """Ensure adjacent ranges have at least `overlap` characters of overlap."""
    if not ranges:
        return ranges
    coalesced = [ranges[0]]
    for s, e in ranges[1:]:
        ps, pe = coalesced[-1]
        if s - pe < overlap:
            s = max(ps, pe - overlap)
        coalesced.append((s, e))
    return coalesced

def build_chunks(text: str) -> List[str]:
    """Return list of text chunks using separator strategy, falling back to size-based."""
    ranges = chunks_by_separators(text)
    if not ranges:
        ranges = chunks_by_size(text)
    return [text[s:e] for (s, e) in ranges]

# ======================
# Main
# ======================
def main():
    # Load processing log
    if not PROCESSING_LOG_CSV.exists():
        print("No processing log found; nothing to do.")
        return

    with PROCESSING_LOG_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if LOG_FILENAME_COL not in fieldnames or LOG_STATUS_COL not in fieldnames:
            raise ValueError(f"Processing log must have columns: {LOG_FILENAME_COL}, {LOG_STATUS_COL}")
        log_rows = list(reader)

    # Build map of OCR .txt files by stem (case-insensitive)
    txt_files_map = {p.stem.lower(): p for p in OCR_OUTPUT_DIR.rglob("*.txt")}

    # Worklist: rows pending + with a matching .txt
    work_indices = []
    for i, r in enumerate(log_rows):
        if str(r.get(LOG_STATUS_COL, "")).strip().lower() != STATUS_PENDING:
            continue
        base = Path(r.get(LOG_FILENAME_COL, "")).stem.lower()
        if base and base in txt_files_map:
            work_indices.append(i)

    if not work_indices:
        print("No pending files found with matching OCR .txt files.")
        return

    # Apply MAX_FILES
    if MAX_FILES is not None:
        work_indices = work_indices[:MAX_FILES]

    system_msg = load_system_message(SYSTEM_MESSAGE_FILE)
    client = make_client()

    total_files = 0
    total_rows = 0
    start = time.time()

    for idx in work_indices:
        if MAX_MINUTES is not None and (time.time() - start) > MAX_MINUTES * 60:
            print(f"Reached MAX_MINUTES={MAX_MINUTES}, stopping early.")
            break

        row = log_rows[idx]
        base = Path(row[LOG_FILENAME_COL]).stem.lower()
        txt_path = txt_files_map.get(base)
        if not txt_path:
            row[LOG_STATUS_COL] = STATUS_FAILED
            print(f"[failed] {row[LOG_FILENAME_COL]} (missing OCR .txt)")
            continue

        payor = derive_payor(OCR_OUTPUT_DIR, txt_path)

        try:
            ocr_text = read_text(txt_path)
            parts = build_chunks(ocr_text)
            if len(parts) > 1:
                print(f"[chunking] {row[LOG_FILENAME_COL]} -> {len(parts)} chunks")

            file_failed = False
            file_rows = 0

            for part_idx, part in enumerate(parts, 1):
                # Time budget check before each chunk
                if MAX_MINUTES is not None and (time.time() - start) > MAX_MINUTES * 60:
                    print(f"Reached MAX_MINUTES={MAX_MINUTES}, stopping early mid-file.")
                    file_failed = True
                    break

                try:
                    raw = call_gpt_with_retries(client, system_msg, part)
                    out_rows = []
                    for jline in sanitize_jsonl(raw):
                        try:
                            obj = json.loads(jline)
                        except json.JSONDecodeError:
                            continue
                        rec = {k: obj.get(k, None) for k in OUTPUT_FIELDS if k not in ("filename", "payor")}
                        rec["filename"] = row[LOG_FILENAME_COL]
                        rec["payor"] = payor
                        out_rows.append(rec)

                    append_rows(out_rows, PARSED_RESULTS_CSV)
                    file_rows += len(out_rows)
                    print(f"[chunk {part_idx}/{len(parts)}] {row[LOG_FILENAME_COL]} -> {len(out_rows)} row(s)")

                except Exception as e:
                    file_failed = True
                    print(f"[failed chunk {part_idx}/{len(parts)}] {row[LOG_FILENAME_COL]} — {type(e).__name__}")

            if not file_failed:
                row[LOG_STATUS_COL] = STATUS_PARSED
                total_files += 1
                total_rows += file_rows
                print(f"[parsed] {row[LOG_FILENAME_COL]} -> total {file_rows} row(s)")
            else:
                row[LOG_STATUS_COL] = STATUS_FAILED
                print(f"[failed] {row[LOG_FILENAME_COL]} (one or more chunks failed)")

        except Exception as e:
            row[LOG_STATUS_COL] = STATUS_FAILED
            print(f"[failed] {row.get(LOG_FILENAME_COL, '<unknown>')} — {type(e).__name__}")

    # Rewrite the processing log with updated statuses
    PROCESSING_LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    with PROCESSING_LOG_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"Done. Files parsed: {total_files}, rows appended: {total_rows}")
    print(f"Results -> {PARSED_RESULTS_CSV}")
    print(f"Updated log -> {PROCESSING_LOG_CSV}")

if __name__ == "__main__":
    main()
