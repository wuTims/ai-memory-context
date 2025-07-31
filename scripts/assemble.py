#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
from retrieve import get_relevant
import sys

# ── Locate base paths ──
BASE_DIR    = os.path.expanduser('~/ai-memory-context')
MEMORY_DIR  = os.path.join(BASE_DIR, 'memory')
INDICES_DIR = os.path.join(BASE_DIR, 'indices')

# ── Load all records ──
docs = []
docs_path = os.path.join(INDICES_DIR, 'docs.npy')
if os.path.exists(docs_path):
    docs = np.load(docs_path, allow_pickle=True).tolist()

# ── Find latest legacy-summary ──
mega = None
for fn in os.listdir(MEMORY_DIR):
    if fn.endswith('_legacy-summary.json'):
        mega = json.load(open(os.path.join(MEMORY_DIR, fn)))
        break

# ── Find latest style-mega-summary ──
style_mega = None
for fn in os.listdir(MEMORY_DIR):
    if fn.endswith('style-mega-summary.json'):
        style_mega = json.load(open(os.path.join(MEMORY_DIR, fn)))
        break

# ── Find latest cardano-mega-summary ──
cardano_mega = None
for fn in os.listdir(MEMORY_DIR):
    if fn.endswith('cardano-mega-summary.json'):
        cardano_mega = json.load(open(os.path.join(MEMORY_DIR, fn)))
        break

def assemble(user_q: str) -> str:
    parts = ["SYSTEM: You are the style-memory assistant with the following context:\n"]

    # 1. legacy mega-summary
    if mega:
        parts.append("## Mega-Summary\n")
        for item in mega.get('content', []):
            parts.append(f"• {item}\n")
        parts.append("\n")

    # 2. core style principles
    if style_mega:
        parts.append("## Core Style Principles\n")
        for item in style_mega.get('content', []):
            parts.append(f"• {item}\n")
        parts.append("\n")

    # 3. Cardano development principles
    if cardano_mega:
        parts.append("## Cardano Development Principles\n")
        for item in cardano_mega.get('content', []):
            parts.append(f"• {item}\n")
        parts.append("\n")

    # 4. recent style records
    recent = sorted(docs, key=lambda r: r.get('date', ''), reverse=True)[:10]
    parts.append("## Recent Style Records\n")
    parts.append(json.dumps(recent, indent=2))
    parts.append("\n\n")

    # 5. relevant records (with optional tag filter)
    rel = get_relevant(user_q, k=9, filter_tags=args.filter_tags)
    parts.append("## Relevant Records\n")
    parts.append(json.dumps(rel, indent=2))
    parts.append("\n\n")

    # 6. user question
    parts.append("USER:\n")
    parts.append(user_q + "\n")

    return "".join(parts)

if __name__ == "__main__":
    # ── Parse CLI arguments ──
    parser = argparse.ArgumentParser(
        description="Assemble AI context with optional tag-filtered memory"
    )
    parser.add_argument(
        "--filter-tags", "-t",
        action="append",
        help="Only include memory records whose tags match one of these",
        default=None
    )
    parser.add_argument(
        "prompt",
        help="User input to assemble context for"
    )
    args = parser.parse_args()

    sys.stdout.write(assemble(args.prompt))
