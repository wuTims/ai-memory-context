import os
import json
import datetime
from collections import Counter

# Summarize records older than THRESHOLD_DAYS into a “mega-summary” entry.

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MEMORY_DIR = os.path.join(BASE_DIR, 'memory')
THRESHOLD_DAYS = 30

# Optional Llama model path via env var
llm = None
llm_model_path = os.environ.get('LLAMA_MODEL_PATH')
if llm_model_path:
    try:
        from llama_cpp import Llama  # type: ignore
        if os.path.exists(llm_model_path):
            llm = Llama(model_path=llm_model_path)
    except Exception:
        llm = None

now = datetime.datetime.now(datetime.timezone.utc)
old_descriptions = []
old_records = []

for fn in os.listdir(MEMORY_DIR):
    if not fn.endswith('.json'):
        continue
    path = os.path.join(MEMORY_DIR, fn)
    try:
        rec = json.load(open(path, 'r'))
    except Exception:
        continue
    date_str = rec.get('date')
    if not date_str:
        continue
    try:
        rec_dt = datetime.datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        if rec_dt.tzinfo is None:
            rec_dt = rec_dt.replace(tzinfo=datetime.timezone.utc)
        else:
            rec_dt = rec_dt.astimezone(datetime.timezone.utc)
    except Exception:
        continue
    if rec.get('type') == 'mega-summary':
        continue
    if (now - rec_dt).days >= THRESHOLD_DAYS:
        old_descriptions.append(rec.get('description','') or rec.get('content',''))
        old_records.append(rec)

if not old_descriptions:
    exit(0)

# Try LLM first
bullets = []
if llm:
    prompt = "Summarize these style rules into 3 concise bullet points:\n" + "\n".join(old_descriptions)
    try:
        resp = llm(prompt, max_tokens=256)
        text = resp.get('choices', [{}])[0].get('text','')
        bullets = [line.strip(" -•") for line in text.split('\n') if line.strip()][:3]
    except Exception:
        bullets = []

# Fallback heuristic
if not bullets:
    stop_words = set(['the','and','of','to','a','an','in','for','on','with','is','this'])
    words = []
    for d in old_descriptions:
        for w in d.lower().split():
            w_clean = ''.join(ch for ch in w if ch.isalnum())
            if w_clean and w_clean not in stop_words:
                words.append(w_clean)
    common = [w for w,_ in Counter(words).most_common(5)]
    bullets.append(f"{len(old_records)} archived records summarised covering {', '.join(common[:3])}.")
    if len(common) > 3:
        bullets.append(f"Additional themes include: {', '.join(common[3:])}.")
    types = Counter([r.get('type','unknown') for r in old_records])
    bullets.append("Record types: " + ", ".join(f"{cnt} {typ}" for typ, cnt in types.items()) + ".")

summary_id = f"{now.date()}_legacy-summary"
summary = {
    "id": summary_id,
    "type": "mega-summary",
    "content": bullets,
    "date": now.replace(microsecond=0).isoformat() + 'Z'
}
with open(os.path.join(MEMORY_DIR, f"{summary_id}.json"), 'w') as f_out:
    json.dump(summary, f_out, indent=2)
