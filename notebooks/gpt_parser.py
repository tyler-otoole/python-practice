#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Overpayment Letter Parsing — GPT-4.1 pass on OCR text files (Azure OpenAI)
- Processes ONLY processing_log.csv rows with parsed_status == "pending"
  AND where ocr_output/**/basename.txt exists (case-insensitive basename match).
- Splits each document into context-aware table chunks:
    [CONTEXT] ... (all narrative, with ALL other tables removed)
    [CURRENT_TABLE] ... (exactly one table block, or a slice of it if very large)
- No-table fallback: if a letter has no tables, the entire letter is promoted to CURRENT_TABLE.
- Calls Azure GPT-4.1 per chunk; appends JSONL rows to data/parsed_results.csv.
- Marks file "parsed" only if ALL chunks succeed; otherwise "failed".
- Run controls: MAX_FILES, MAX_MINUTES, MAX_RETRIES.

Adjust the Azure constants below if you're hard-coding.
"""

import csv
import json
import os
import re
import time
from pathlib import Path
from typing import List, Tuple, Iterable
from openai import AzureOpenAI

# ======================
# Paths / Config
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
    "filename",   # original PDF name from processing log
    "payor",      # derived from OCR_OUTPUT_DIR/<PAYOR>/...
]

# ======================
# Chunking controls
# ======================
# Primary sentinel(s) from your OCR pipeline. You said "BEING_TABLE"; include "BEGIN_TABLE" defensively.
TABLE_SENTINEL_PATTERN = re.compile(r"(?im)^\s*(?:BEING_TABLE|BEGIN_TABLE)\b.*$")

# Size limits (fallback or giant tables). Roughly ~4 chars/token; 12k chars ~3k tokens.
MAX_CHARS_PER_CHUNK = 12000
CHUNK_OVERLAP_CHARS = 800

# ======================
# Azure OpenAI (hard-code or swap to env/ dotenv later)
# ======================
AZURE_OPENAI_ENDPOINT = "https://my-overpayment-parser.openai.azure.com/"
AZURE_OPENAI_API_KEY = "YOUR_KEY_HERE"
AZURE_OPENAI_API_VERSION = "2024-10-21"
AZURE_OPENAI_DEPLOYMENT = "gpt-4.1-overpayment"
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 4000  # request near-maximum per response

def make_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )

# ======================
# Small helpers
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

def sanitize_jsonl(raw: str) -> Iterable[str]:
    """Keep only lines that look like JSON objects; strip code fences / trailing commas."""
    for line in raw.splitlines():
        s = line.strip().strip("`").strip()
        if not s:
            continue
        if s.endswith(","):
            s = s[:-1]
        if s.startswith("{") and s.endswith("}"):
            yield s

def append_rows(rows: List[dict], csv_path: Path):
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerows(rows)

# ======================
# GPT call
# ======================
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
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return call_gpt(client, system_msg, user_text)
        except Exception as e:
            last_err = e
            print(f"[retry {attempt}/{MAX_RETRIES}] GPT call failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)  # simple backoff
    raise last_err if last_err else RuntimeError("Unknown GPT error")

# ======================
# Chunk building (context + single table)
# ======================
def locate_table_blocks(text: str) -> List[Tuple[int, int]]:
    """
    Return list of (start_idx, end_idx) for each table block.
    A table starts at a TABLE_SENTINEL line and ends at the next sentinel or EOF.
    """
    starts = [m.start() for m in TABLE_SENTINEL_PATTERN.finditer(text)]
    if not starts:
        return []
    blocks: List[Tuple[int, int]] = []
    for i, s in enumerate(starts):
        e = starts[i+1] if i+1 < len(starts) else len(text)
        blocks.append((s, e))
    return blocks

def strip_blocks(text: str, blocks: List[Tuple[int, int]], up_to: int = None, after: Tuple[int, int] = None) -> str:
    """
    Remove specified table blocks from a slice of text.
    - If up_to is set, keep only text from [0:up_to], then strip any table blocks fully within that slice.
    - If after is set to (s,e), you can pass a slice after that block (e:next_start) and strip tables within it.
    """
    if up_to is not None:
        slice_text = text[:up_to]
        slice_blocks = [(s, e) for (s, e) in blocks if e <= up_to]
    elif after is not None:
        s0, e0 = after
        # determine next start to bound the slice
        # caller is expected to slice text[e0:next_start]
        slice_text = text
        slice_blocks = []  # we assume caller pre-sliced to not include complete tables
    else:
        slice_text = text
        slice_blocks = blocks

    if not slice_blocks:
        return slice_text

    out = []
    pos = 0
    for s, e in slice_blocks:
        if pos < s:
            out.append(slice_text[pos:s])
        pos = e
    if pos < len(slice_text):
        out.append(slice_text[pos:])
    return "".join(out)

def split_block_by_size(text: str, start: int, end: int) -> List[Tuple[int, int]]:
    """Split a big block [start:end] into size-bounded windows with overlap."""
    ranges: List[Tuple[int, int]] = []
    n = end - start
    offset = 0
    while offset < n:
        e = min(offset + MAX_CHARS_PER_CHUNK, n)
        ranges.append((start + offset, start + e))
        if e >= n:
            break
        offset = max(offset, e - CHUNK_OVERLAP_CHARS)
    return ranges

def build_contextual_chunks(text: str) -> List[str]:
    """
    Yield chunks with:
      [CONTEXT]    -> narrative only (all other tables removed)
      [CURRENT_TABLE] -> exactly one table (or a size-slice of it)
    No-table fallback: the entire letter goes to CURRENT_TABLE (may also be split by size).
    """
    blocks = locate_table_blocks(text)

    # No tables: promote whole letter to CURRENT_TABLE (split by size if needed)
    if not blocks:
        # size-split the entire letter if necessary
        whole_ranges = split_block_by_size(text, 0, len(text)) if len(text) > MAX_CHARS_PER_CHUNK else [(0, len(text))]
        chunks: List[str] = []
        for s, e in whole_ranges:
            chunk = f"[CONTEXT]\n\n[CURRENT_TABLE]\n{text[s:e]}\n\n[END]"
            chunks.append(chunk)
        return chunks

    chunks: List[str] = []

    for idx, (s, e) in enumerate(blocks):
        # CONTEXT for this table = (0 -> s) with all earlier tables stripped,
        # plus after-table narrative (e -> next_start) with tables stripped (none fully in that slice).
        pre_context = strip_blocks(text, blocks, up_to=s)
        next_start = blocks[idx + 1][0] if (idx + 1) < len(blocks) else len(text)
        after_slice = text[e:next_start]
        # after_slice shouldn't contain full tables by construction, so no need to strip again
        context = (pre_context + "\n" + after_slice).strip()

        # CURRENT_TABLE = this table block (may need size-splitting)
        table_ranges = [(s, e)]
        if (e - s) > MAX_CHARS_PER_CHUNK:
            table_ranges = split_block_by_size(text, s, e)

        for ts, te in table_ranges:
            chunk = f"[CONTEXT]\n{context}\n\n[CURRENT_TABLE]\n{text[ts:te]}\n\n[END]"
            chunks.append(chunk)

    return chunks

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
            parts = build_contextual_chunks(ocr_text)
            if len(parts) > 1:
                print(f"[chunking] {row[LOG_FILENAME_COL]} -> {len(parts)} chunk(s)")

            file_failed = False
            file_rows = 0

            for part_idx, part in enumerate(parts, 1):
                if MAX_MINUTES is not None and (time.time() - start) > MAX_MINUTES * 60:
                    print(f"Reached MAX_MINUTES={MAX_MINUTES}, stopping early mid-file.")
                    file_failed = True
                    break

                try:
                    raw = call_gpt_with_retries(client, system_msg, part)
                    out_rows: List[dict] = []
                    for jline in sanitize_jsonl(raw):
                        try:
                            obj = json.loads(jline)
                        except json.JSONDecodeError:
                            continue
                        rec = {k: obj.get(k, None) for k in OUTPUT_FIELDS if k not in ("filename", "payor")}
                        rec["filename"] = row[LOG_FILENAME_COL]  # original PDF name from log
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
