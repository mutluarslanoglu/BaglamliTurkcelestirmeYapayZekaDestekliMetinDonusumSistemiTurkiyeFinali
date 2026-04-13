from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

# CORS
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(BASE_DIR, "user_profile.db")

# -----------------------------
# Dosya okuma
# -----------------------------

def load_lines(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f.readlines()
            if line.strip() and not line.strip().startswith("#")
        ]

def load_json(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_token(tok: str) -> str:
    return tok.lower()

def preserve_casing(original: str, suggestion: str) -> str:
    if original.isupper():
        return suggestion.upper()
    if original[:1].isupper():
        return suggestion[:1].upper() + suggestion[1:]
    return suggestion

def is_all_caps(token: str) -> bool:
    return token.isupper() and len(token) >= 2

def looks_like_proper_noun(original_token: str, token_index_in_sentence: int) -> bool:
    if token_index_in_sentence == 0:
        return False
    return original_token[:1].isupper()

# -----------------------------
# Basit TR ek ayrıştırma (heuristik)
# -----------------------------
TR_SUFFIX_RE = re.compile(
    r"(?P<root>[a-zçğıöşü]+)"
    r"(?P<suffix>(?:"
    r"(?:lar|ler)"
    r"|(?:ım|im|um|üm|m)"
    r"|(?:ın|in|un|ün|n)"
    r"|(?:ı|i|u|ü)"
    r"|(?:a|e)"
    r"|(?:da|de|ta|te)"
    r"|(?:dan|den|tan|ten)"
    r"|(?:ya|ye)"
    r"|(?:ki)"
    r"|(?:dır|dir|dur|dür|tır|tir|tur|tür)"
    r"|(?:mış|miş|muş|müş)"
    r"|(?:acak|ecek)"
    r"|(?:yı|yi|yu|yü)"
    r"|(?:yla|yle)"
    r")*)$",
    re.IGNORECASE
)

def split_root_suffix(norm: str) -> Tuple[str, str]:
    m = TR_SUFFIX_RE.match(norm)
    if not m:
        return norm, ""
    return m.group("root"), m.group("suffix") or ""

# -----------------------------
# SQLite öğrenme (WAL + timeout + thread-safe bağlantı)
# -----------------------------
def get_con() -> sqlite3.Connection:
    con = sqlite3.connect(
        DB_PATH,
        timeout=5.0,
        check_same_thread=False,
    )
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def db_init():
    with get_con() as con:
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS prefs(
                user_id TEXT NOT NULL,
                foreign_term TEXT NOT NULL,
                suggestion TEXT NOT NULL,
                context_tag TEXT NOT NULL,
                score INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(user_id, foreign_term, suggestion, context_tag)
            )
        """)
        con.commit()

def db_add_score(user_id: str, foreign_term: str, suggestion: str, context_tag: str, delta: int):
    with get_con() as con:
        cur = con.cursor()
        cur.execute("""
            INSERT INTO prefs(user_id, foreign_term, suggestion, context_tag, score)
            VALUES(?,?,?,?,?)
            ON CONFLICT(user_id, foreign_term, suggestion, context_tag)
            DO UPDATE SET score = score + excluded.score
        """, (user_id, foreign_term, suggestion, context_tag, delta))
        con.commit()

def db_get_scores(user_id: str, foreign_term: str, context_tag: str) -> Dict[str, int]:
    with get_con() as con:
        cur = con.cursor()
        cur.execute("""
            SELECT suggestion, score FROM prefs
            WHERE user_id=? AND foreign_term=? AND context_tag=?
        """, (user_id, foreign_term, context_tag))
        rows = cur.fetchall()
    return {s: int(sc) for s, sc in rows}

# -----------------------------
# Uygulama verileri
# -----------------------------
SUGGESTIONS: Dict[str, List[str]] = load_json(os.path.join(DATA_DIR, "suggestions_2000.json"))

FOREIGN_TERMS_RAW = load_lines(os.path.join(DATA_DIR, "foreign_terms.txt"))
FOREIGN_TERMS = {t.lower() for t in FOREIGN_TERMS_RAW}

WHITELIST_RAW = load_lines(os.path.join(DATA_DIR, "whitelist.txt"))
WHITELIST = set(WHITELIST_RAW)
WHITELIST_NORM = {w.lower() for w in WHITELIST_RAW}

PHRASES = sorted([t for t in FOREIGN_TERMS if " " in t], key=len, reverse=True)

COMMON_LOANWORDS = {
    "proje", "rapor", "analiz", "model", "metod", "metodoloji",
    "test", "grafik", "tablo", "form", "format", "sistem"
}

CODE_LIKE_RE = re.compile(r"```.*?```", re.DOTALL)
URL_RE = re.compile(r"https?://\S+")
EMAIL_RE = re.compile(r"\b\S+@\S+\.\S+\b")

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
TOKEN_RE = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşü]+(?:['’\-][A-Za-zÇĞİÖŞÜçğıöşü]+)?")

WORDCHAR_CLASS = r"A-Za-zÇĞİÖŞÜçğıöşü0-9_"
def phrase_pattern(ph: str) -> re.Pattern:
    return re.compile(rf"(?<![{WORDCHAR_CLASS}]){re.escape(ph)}(?![{WORDCHAR_CLASS}])", re.IGNORECASE)

@dataclass
class Candidate:
    id: str
    original: str
    foreign_norm: str
    start: int
    end: int
    context: str
    root: str
    suffix: str

def protect_spans(text: str) -> List[Tuple[int, int]]:
    spans = []
    for m in CODE_LIKE_RE.finditer(text):
        spans.append((m.start(), m.end()))
    for m in URL_RE.finditer(text):
        spans.append((m.start(), m.end()))
    for m in EMAIL_RE.finditer(text):
        spans.append((m.start(), m.end()))
    spans.sort()
    return spans

def overlaps(a: int, b: int, x: int, y: int) -> bool:
    return a < y and b > x

def in_protected(a: int, b: int, protected: List[Tuple[int, int]]) -> bool:
    for x, y in protected:
        if overlaps(a, b, x, y):
            return True
    return False

def level_allows(term_norm: str, level: str) -> bool:
    if level == "strict":
        return True
    if level == "balanced":
        return term_norm not in COMMON_LOANWORDS
    is_ascii = term_norm.isascii()
    return (term_norm in SUGGESTIONS) or (is_ascii and len(term_norm) >= 4)

def split_sentences_with_spans(text: str) -> List[Tuple[str, int, int]]:
    if not text:
        return []
    parts = SENT_SPLIT.split(text)
    spans: List[Tuple[str, int, int]] = []
    pos = 0
    for p in parts:
        p = p.strip()
        if not p:
            continue
        start = text.find(p, pos)
        if start == -1:
            start = pos
        end = start + len(p)
        spans.append((p, start, end))
        pos = end
    return spans

def get_sentence_context(text: str, idx: int) -> str:
    spans = split_sentences_with_spans(text)
    for s, a, b in spans:
        if a <= idx <= b:
            return s.strip()
    return text.strip()

def detect_candidates(text: str, level: str) -> List[Candidate]:
    protected = protect_spans(text)
    cands: List[Candidate] = []

    for ph in PHRASES:
        if not level_allows(ph, level):
            continue
        if any(w.lower() in WHITELIST_NORM for w in ph.split()):
            continue

        pat = phrase_pattern(ph)
        for m in pat.finditer(text):
            a, b = m.start(), m.end()
            if in_protected(a, b, protected):
                continue
            cid = f"ph:{a}:{b}"
            cands.append(Candidate(
                id=cid,
                original=text[a:b],
                foreign_norm=ph.lower(),
                start=a,
                end=b,
                context=get_sentence_context(text, a),
                root=ph.lower(),
                suffix=""
            ))

    sentences = split_sentences_with_spans(text.strip())
    for s, base_a, _base_b in sentences:
        tokens = list(TOKEN_RE.finditer(s))
        for ti, m in enumerate(tokens):
            original = m.group(0)
            a, b = base_a + m.start(), base_a + m.end()
            if in_protected(a, b, protected):
                continue

            norm = normalize_token(original)

            if original in WHITELIST or norm in WHITELIST_NORM:
                continue
            if is_all_caps(original):
                continue
            if looks_like_proper_noun(original, ti):
                continue

            root, suffix = split_root_suffix(norm)

            hit = None
            if norm in FOREIGN_TERMS:
                hit = norm
            elif root in FOREIGN_TERMS:
                hit = root

            if not hit:
                continue
            if not level_allows(hit, level):
                continue

            cid = f"w:{a}:{b}"
            cands.append(Candidate(
                id=cid,
                original=original,
                foreign_norm=hit,
                start=a,
                end=b,
                context=s,
                root=root,
                suffix=suffix
            ))

    cands.sort(key=lambda c: (c.start, -(c.end - c.start)))
    filtered: List[Candidate] = []
    last_end = -1
    for c in cands:
        if c.start < last_end:
            continue
        filtered.append(c)
        last_end = c.end
    return filtered

def rank_suggestions(user_id: str, foreign_norm: str, base_suggestions: List[str], context_tag: str) -> List[Dict[str, Any]]:
    scores = db_get_scores(user_id, foreign_norm, context_tag)
    items = [{"suggestion": s, "score": scores.get(s, 0)} for s in base_suggestions]
    items.sort(key=lambda x: x["score"], reverse=True)
    return items

def apply_replacements(text: str, replacements: List[Dict]) -> str:
    reps = sorted(replacements, key=lambda r: r["start"], reverse=True)
    out = text
    for r in reps:
        out = out[:r["start"]] + r["new"] + out[r["end"]:]
    return out

def attach_original_suffix(original: str, candidate_suffix: str, new_root: str) -> str:
    if not candidate_suffix:
        return new_root
    suf_len = len(candidate_suffix)
    if suf_len <= 0 or len(original) <= suf_len:
        return new_root
    original_suffix_text = original[-suf_len:]
    return new_root + original_suffix_text

# -----------------------------
# API Şemaları
# -----------------------------
class AnalyzeRequest(BaseModel):
    user_id: str = "default"
    text: str
    context_tag: str = "akademik"
    level: str = "balanced"

class Choice(BaseModel):
    candidate_id: str
    chosen: Optional[str] = None
    rejected: List[str] = []

class ApplyRequest(BaseModel):
    user_id: str = "default"
    text: str
    context_tag: str = "akademik"
    level: str = "balanced"
    choices: List[Choice]

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title="Bağlam Duyarlı Türkçeleştirme Asistanı")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

db_init()

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    cands = detect_candidates(req.text, req.level)
    results = []

    for c in cands:
        base = SUGGESTIONS.get(c.foreign_norm, [])
        ranked = rank_suggestions(req.user_id, c.foreign_norm, base, req.context_tag)

        # Orijinal kelime de öneri olsun
        # UI'da label: "(Orijinal kalsın) <kelime>" görünsün
        orig = c.original.strip()
        if orig:
            exists = {x.get("suggestion") for x in ranked}
            if orig not in exists:
                ranked = [{"suggestion": orig, "label": f"(Orijinal kalsın) {orig}", "score": 0}] + ranked

        results.append({
            "id": c.id,
            "original": c.original,
            "foreign_norm": c.foreign_norm,
            "start": c.start,
            "end": c.end,
            "context": c.context,
            "root": c.root,
            "suffix": c.suffix,
            "suggestions": ranked
        })

    report = {
        "candidates_found": len(results),
        "unique_foreign_terms": len(set(r["foreign_norm"] for r in results)),
        "level": req.level
    }
    return {"items": results, "report": report}

@app.post("/apply")
def apply(req: ApplyRequest):
    analyzed = analyze(AnalyzeRequest(
        user_id=req.user_id,
        text=req.text,
        context_tag=req.context_tag,
        level=req.level
    ))
    items = {it["id"]: it for it in analyzed["items"]}

    replacements = []

    for ch in req.choices:
        it = items.get(ch.candidate_id)
        if not it:
            continue

        foreign = it["foreign_norm"]
        original = it["original"]
        suffix = it.get("suffix", "") or ""

        for r in ch.rejected:
            if r:
                db_add_score(req.user_id, foreign, r, req.context_tag, -1)

        if ch.chosen:
            db_add_score(req.user_id, foreign, ch.chosen, req.context_tag, +2)

            original_root_text = original[:-len(suffix)] if suffix and len(original) > len(suffix) else original
            new_root = preserve_casing(original_root_text, ch.chosen)
            new_word = attach_original_suffix(original, suffix, new_root)

            replacements.append({"start": it["start"], "end": it["end"], "new": new_word})

    new_text = apply_replacements(req.text, replacements)

    return {
        "new_text": new_text,
        "applied_count": len(replacements),
        "report": analyzed["report"]
    }

@app.get("/")
def root():
    return FileResponse(os.path.join(BASE_DIR, "index2.html"))

@app.get("/health")
def health():
    return {"ok": True}