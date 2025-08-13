#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Overpayment Letter Parsing — Azure GPT-4.1 with robust BEGIN/END TABLE chunking.

- Processes ONLY processing_log rows where parsed_status == "pending"
  AND a matching OCR .txt exists (basename match to CSV `filename`).
- Splits each letter into chunks: one per table, bounded by "BEGIN TABLE" ... "END TABLE".
  * [CONTEXT]       narrative text with ALL tables removed (intro + between-table boilerplate)
  * [CURRENT_TABLE] exactly ONE table's rows (BEGIN/END lines removed)
- If a table is very large, we split it by size to avoid output caps.
- If NO tables exist, the whole letter is treated as CURRENT_TABLE (split by size if needed).
- On success: set parsed_status="parsed". If any chunk fails or time budget exceeded: "failed".
"""

import csv
import json
import os
import re
import time
from pathlib import Path
from typing import List, Tuple, Iterable, Optional
from openai import AzureOpenAI

# ======================
# CONFIG
# ======================
OCR_OUTPUT_DIR = Path("ocr_output")
SYSTEM_MESSAGE_FILE = Path("prompts/system_message.txt")
PARSED_RESULTS_CSV = Path("data/parsed_results.csv")
PROCESSING_LOG_CSV = Path("data/processing_log.csv")

# Processing log fields
LOG_FILENAME_COL = "filename"      # e.g., image19857.pdf
LOG_STATUS_COL = "parsed_status"   # "pending" | "parsed" | "failed"
STATUS_PENDING = "pending"
STATUS_PARSED = "parsed"
STATUS_FAILED = "failed"

# Batch / run control
MAX_FILES = 10        # None for no limit
MAX_MINUTES = 10      # None for no time limit
MAX_RETRIES = 2       # per-chunk GPT retries with simple backoff

# Output schema + enrichments
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
    "filename",  # original PDF name
    "payor",     # derived from ocr_output/<PAYOR>/...
]

# ===== Table sentinel configuration =====
# EXACT sentinels are "BEGIN TABLE" and "END TABLE" (case-insensitive).
START_PAT = re.compile(r"(?im)^\s*BEGIN\s+TABLE\s*$")
END_PAT   = re.compile(r"(?im)^\s*END\s+TABLE\s*$")

# Size limits (~4 chars/token; 12k chars ≈ 3k tokens)
MAX_CHARS_PER_CHUNK = 12000
CHUNK_OVERLAP_CHARS = 800

# ===== Debugging aid =====
DEBUG_DUMP_CHUNKS = False
DEBUG_DIR = Path("debug_chunks")  # will contain <pdfname>__chunkNN.txt

# ======================
# Azure OpenAI (hard-code for now)
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
    try:
        rel = file_path.resolve().relative_to(ocr_root.resolve())
        return rel.parts[0] if len(rel.parts) > 1 else ""
    except Exception:
        return ""

def sanitize_jsonl(raw: str) -> Iterable[str]:
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
                time.sleep(2 * attempt)
    raise last_err if last_err else RuntimeError("Unknown GPT error")

# ======================
# Table detection & chunking
# ======================
def _line_indices(pattern: re.Pattern, text: str) -> List[int]:
    return [m.start() for m in pattern.finditer(text)]

def _strip_lines(pattern: re.Pattern, text: str) -> str:
    """Remove all lines that match a pattern (to strip BEGIN/END markers)."""
    keep = []
    for line in text.splitlines(keepends=True):
        if not pattern.match(line):
            keep.append(line)
    return "".join(keep)

def locate_table_blocks(text: str) -> List[Tuple[int, int]]:
    """
    Return [(start,end)] for each table: from BEGIN TABLE line to END TABLE line.
    If a BEGIN has no END after it, we terminate at EOF.
    If an END appears before any BEGIN (weird OCR), we ignore it.
    """
    starts = _line_indices(START_PAT, text)
    ends = _line_indices(END_PAT, text)
    if not starts:
        return []

    blocks: List[Tuple[int, int]] = []
    e_idx = 0
    for s in starts:
        # advance e_idx to the first END >= s
        while e_idx < len(ends) and ends[e_idx] < s:
            e_idx += 1
        if e_idx < len(ends):
            e = ends[e_idx]
            e_idx += 1
        else:
            e = len(text)
        blocks.append((s, e))
    return blocks

def strip_blocks(text: str, blocks: List[Tuple[int, int]], end_limit: Optional[int] = None) -> str:
    """
    Remove all table blocks from text (optionally up to end_limit).
    Produces narrative-only text for [CONTEXT].
    """
    out = []
    pos = 0
    limit = len(text) if end_limit is None else min(end_limit, len(text))
    for s, e in blocks:
        if s >= limit:
            break
        if pos < s:
            out.append(text[pos:s])   # keep narrative before table
        pos = max(pos, min(e, limit)) # skip table region
    if pos < limit:
        out.append(text[pos:limit])
    return "".join(out)

def split_range_by_size(start: int, end: int) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    n = end - start
    off = 0
    while off < n:
        e = min(off + MAX_CHARS_PER_CHUNK, n)
        ranges.append((start + off, start + e))
        if e >= n:
            break
        off = max(off, e - CHUNK_OVERLAP_CHARS)
    return ranges

def build_chunks(text: str) -> List[str]:
    """
    Build chunks with:
      [CONTEXT]       -> narrative only, all tables removed
      [CURRENT_TABLE] -> exactly one table (BEGIN/END stripped), size-split if necessary
    If no tables -> entire text promoted to CURRENT_TABLE (may be size-split).
    """
    blocks = locate_table_blocks(text)

    # No tables: promote whole letter into CURRENT_TABLE
    if not blocks:
        ranges = split_range_by_size(0, len(text)) if len(text) > MAX_CHARS_PER_CHUNK else [(0, len(text))]
        return [f"[CONTEXT]\n\n[CURRENT_TABLE]\n{text[s:e]}\n\n[END]" for s, e in ranges]

    chunks: List[str] = []

    for i, (s, e) in enumerate(blocks):
        next_start = blocks[i+1][0] if i + 1 < len(blocks) else len(text)

        # CONTEXT = narrative text before this table (tables removed) + narrative after table up to next table
        pre_context = strip_blocks(text, blocks, end_limit=s)
        after_context = text[e:next_start]  # by construction contains no complete tables
        context = (pre_context + "\n" + after_context).strip()

        # Table content slice; remove BEGIN/END marker lines
        table_slice = text[s:e]
        table_slice = _strip_lines(START_PAT, table_slice)
        table_slice = _strip_lines(END_PAT, table_slice)

        # Split large table slices
        if (e - s) > MAX_CHARS_PER_CHUNK:
            ranges = split_range_by_size(s, e)
            for rs, re_ in ranges:
                part = text[rs:re_]
                part = _strip_lines(START_PAT, part)
                part = _strip_lines(END_PAT, part)
                chunks.append(f"[CONTEXT]\n{context}\n\n[CURRENT_TABLE]\n{part}\n\n[END]")
        else:
            chunks.append(f"[CONTEXT]\n{context}\n\n[CURRENT_TABLE]\n{table_slice}\n\n[END]")

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

    # Map of OCR .txt files by basename (case-insensitive)
    txt_files = {p.stem.lower(): p for p in OCR_OUTPUT_DIR.rglob("*.txt")}

    # Build worklist
    work_indices = []
    for i, r in enumerate(log_rows):
        if str(r.get(LOG_STATUS_COL, "")).strip().lower() != STATUS_PENDING:
            continue
        base = Path(r.get(LOG_FILENAME_COL, "")).stem.lower()
        if base and base in txt_files:
            work_indices.append(i)

    if not work_indices:
        print("No pending files found with matching OCR .txt files.")
        return

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
        txt_path = txt_files.get(base)
        if not txt_path:
            row[LOG_STATUS_COL] = STATUS_FAILED
            print(f"[failed] {row[LOG_FILENAME_COL]} (missing OCR .txt)")
            continue

        payor = derive_payor(OCR_OUTPUT_DIR, txt_path)

        try:
            ocr_text = read_text(txt_path)
            parts = build_chunks(ocr_text)
            print(f"[chunking] {row[LOG_FILENAME_COL]} -> {len(parts)} chunk(s)")

            if DEBUG_DUMP_CHUNKS:
                DEBUG_DIR.mkdir(parents=True, exist_ok=True)
                for n, c in enumerate(parts, 1):
                    (DEBUG_DIR / f"{Path(row[LOG_FILENAME_COL]).stem}__chunk{n:02d}.txt").write_text(c, encoding="utf-8")

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

    # Rewrite processing log
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
