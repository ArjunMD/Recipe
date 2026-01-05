# app.py
# Run: streamlit run app.py

from __future__ import annotations

import difflib
import html
import json
import re
import uuid
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import streamlit as st

# =============================================================================
# App config / storage locations
# =============================================================================
SCHEMA_VERSION = 6

DATA_DIR = Path("data")
DB_FILE = DATA_DIR / "recipes_db.json"
PHOTOS_DIR = DATA_DIR / "photos"

# Fields that define a “recipe state” snapshot.
VERSION_FIELDS = ("name", "source", "ingredients", "instructions")


# =============================================================================
# Time / ids / filesystem
# =============================================================================
def now_iso() -> str:
    """Current local timestamp (ISO, seconds precision)."""
    return datetime.now().isoformat(timespec="seconds")


def new_id() -> str:
    """Random stable id for recipes/entries/photos/versions."""
    return uuid.uuid4().hex


def ensure_dirs() -> None:
    """Create data dirs if missing."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DB read/write
# =============================================================================
def _empty_db() -> Dict[str, Any]:
    return {
        "version": SCHEMA_VERSION,
        "recipes": {},   # recipe_id -> recipe dict
        "entries": [],   # unified timeline: cook/edit/note/variation
        "photos": [],    # photos attached to cook entries
    }


def load_db() -> Dict[str, Any]:
    """Load DB file, returning a safe default shape on any failure."""
    ensure_dirs()
    if not DB_FILE.exists():
        db = _empty_db()
        save_db(db)
        return db

    try:
        raw = DB_FILE.read_text(encoding="utf-8")
        db = json.loads(raw)
        if not isinstance(db, dict):
            return _empty_db()

        db.setdefault("version", SCHEMA_VERSION)
        db.setdefault("recipes", {})
        db.setdefault("entries", [])
        db.setdefault("photos", [])
        return db
    except Exception:
        return _empty_db()


def save_db(db: Dict[str, Any]) -> None:
    """Atomic-ish write (tmp file then replace)."""
    ensure_dirs()
    tmp = DB_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(db, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(DB_FILE)


# =============================================================================
# Date/time parsing + display
# =============================================================================
def fmt_stamp(x: Any) -> str:
    """
    Display-only: ISO datetime/date (or date/datetime objects) -> "Mon DD, 'YY".
    Leaves unknown formats as-is.
    """
    if x is None:
        return ""

    if isinstance(x, (datetime, date)):
        return x.strftime("%b %d, '%y")

    s = str(x).strip()
    if not s:
        return ""

    try:
        if "T" in s:
            return datetime.fromisoformat(s).strftime("%b %d, '%y")
    except Exception:
        pass

    try:
        return date.fromisoformat(s[:10]).strftime("%b %d, '%y")
    except Exception:
        return s


def _parse_date_only(s: str) -> Optional[date]:
    """Best-effort ISO date extraction from date or datetime strings."""
    s = (s or "").strip()
    if not s:
        return None

    try:
        if "T" in s:
            return datetime.fromisoformat(s).date()
    except Exception:
        pass

    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


def _parse_time_only(s: str) -> time:
    """
    Best-effort time-of-day extraction from ISO datetime strings.
    Fallback is 08:00 (stable ordering for same-day items).
    """
    s = (s or "").strip()
    if not s:
        return time(8, 0, 0)

    try:
        if "T" in s:
            return datetime.fromisoformat(s).time().replace(microsecond=0)
    except Exception:
        pass

    return time(8, 0, 0)


# =============================================================================
# Schema helpers (recipes / variations / entries)
# =============================================================================
def ensure_recipe(r: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a recipe dict in-place and return it."""
    r.setdefault("id", r.get("id") or new_id())
    r.setdefault("name", "")
    r.setdefault("source", "")
    r.setdefault("ingredients", "")
    r.setdefault("instructions", "")
    r.setdefault("created_at", r.get("created_at") or now_iso())
    r.setdefault("updated_at", r.get("updated_at") or r.get("created_at") or now_iso())

    # Original immutable snapshot + edit history snapshots (newest-first)
    r.setdefault("original", None)
    r.setdefault("versions", [])

    # Variations: list[dict]
    r.setdefault("variations", [])
    return r


def ensure_variation(v: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a variation dict in-place and return it."""
    v.setdefault("id", v.get("id") or new_id())
    v.setdefault("title", v.get("title") or v.get("name") or "")
    v.setdefault("text", v.get("text") or "")

    # Auto-title if missing
    if not (v.get("title") or "").strip():
        first = ((v.get("text") or "").strip().splitlines() or ["Variation"])[0].strip()
        v["title"] = first[:80] if first else "Variation"

    v.setdefault("created_at", v.get("created_at") or now_iso())
    v.setdefault("updated_at", v.get("updated_at") or v.get("created_at") or now_iso())
    return v


def _normalized_variations(recipe: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return normalized list[variation] (dicts only)."""
    out: List[Dict[str, Any]] = []
    for item in recipe.get("variations", []) or []:
        if isinstance(item, dict):
            out.append(ensure_variation(item))
    return out


def variation_label(v: Dict[str, Any], n: int = 80) -> str:
    title = (v.get("title") or "").strip() or "Variation"
    return title if len(title) <= n else (title[:n].rstrip() + "…")


def ensure_entry(e: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a timeline entry dict in-place and return it."""
    e.setdefault("id", e.get("id") or new_id())
    e.setdefault("recipe_id", "")
    e.setdefault("type", "note")  # cook | edit | note | variation
    e.setdefault("created_at", e.get("created_at") or now_iso())
    e.setdefault("thoughts", "")

    # cook fields
    e.setdefault("cooked_on", "")
    e.setdefault("cook_notes", "")
    e.setdefault("edited_recipe", False)
    e.setdefault("no_edit_reason", "")
    e.setdefault("associated_version_id", "")
    e.setdefault("variation_id", "")
    e.setdefault("variation_text", "")
    e.setdefault("variation_title", "")
    e.setdefault("variation_action", "")

    # edit fields (legacy/optional)
    e.setdefault("edit_note", "")
    return e


# =============================================================================
# Recipe snapshots / diffs
# =============================================================================
def make_original_snapshot(r: Dict[str, Any]) -> Dict[str, Any]:
    rr = ensure_recipe(dict(r))
    return {
        "saved_at": rr.get("created_at") or now_iso(),
        "name": rr.get("name", ""),
        "source": rr.get("source", ""),
        "ingredients": rr.get("ingredients", ""),
        "instructions": rr.get("instructions", ""),
    }


def make_version_snapshot(
    recipe_before: Dict[str, Any],
    label: str,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Snapshot of the recipe *before* an edit is applied.
    Stored newest-first in recipe["versions"].
    """
    rr = ensure_recipe(dict(recipe_before))
    v: Dict[str, Any] = {
        "id": new_id(),
        "saved_at": now_iso(),
        "label": (label or "Edit"),
    }
    for k in VERSION_FIELDS:
        v[k] = rr.get(k, "")
    if meta:
        v.update(meta)
    return v


def combined_text(obj: Dict[str, Any]) -> str:
    """Ingredients + instructions as one block for diffing."""
    ing = (obj.get("ingredients") or "").rstrip()
    ins = (obj.get("instructions") or "").rstrip()
    return (ing + "\n\n" + ins).strip()


def changes_only(before_text: str, after_text: str, max_lines: Optional[int] = None) -> str:
    """Line-based ndiff, returning only +/- lines."""
    b = (before_text or "").splitlines()
    a = (after_text or "").splitlines()
    out: List[str] = []
    for line in difflib.ndiff(b, a):
        if line.startswith("- "):
            out.append(f"- {line[2:]}")
        elif line.startswith("+ "):
            out.append(f"+ {line[2:]}")
    if not out:
        return ""
    if max_lines is not None and len(out) > max_lines:
        out = out[:max_lines] + ["… (truncated)"]
    return "\n".join(out)


# ---- Inline “tracked changes” (token-level) for original-vs-current ----
_TOKEN_RE = re.compile(r"\w+|\s+|[^\w\s]", re.UNICODE)


def _tokenize(s: str) -> List[str]:
    return _TOKEN_RE.findall(s or "")


def inline_tracked_changes_html(before_text: str, after_text: str) -> str:
    """
    Render AFTER text, with:
      - inserted tokens highlighted green
      - deleted tokens re-inserted in-place highlighted red + struck-through
    """
    bt = _tokenize(before_text or "")
    at = _tokenize(after_text or "")
    sm = difflib.SequenceMatcher(None, bt, at)
    out: List[str] = []

    def emit(tok: str) -> str:
        return html.escape(tok)

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out.extend(emit(t) for t in at[j1:j2])
        elif tag == "insert":
            out.extend(f"<span class='diff-ins-inline'>{emit(t)}</span>" for t in at[j1:j2])
        elif tag == "delete":
            out.extend(f"<span class='diff-del-inline'>{emit(t)}</span>" for t in bt[i1:i2])
        elif tag == "replace":
            out.extend(f"<span class='diff-del-inline'>{emit(t)}</span>" for t in bt[i1:i2])
            out.extend(f"<span class='diff-ins-inline'>{emit(t)}</span>" for t in at[j1:j2])

    return (
        "<div class='diff-inline-box' "
        "style='white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "
        "\"Liberation Mono\", \"Courier New\", monospace; line-height: 1.35;'>"
        + "".join(out)
        + "</div>"
    )


# ---- Compact “snippets” diff (line-aware, token-highlight inside each line) ----
def _render_line_with_mask(tokens: List[str], mask: List[bool], cls: str) -> str:
    out: List[str] = []
    for i, tok in enumerate(tokens):
        t = html.escape(tok)
        out.append(f"<span class='{cls}'>{t}</span>" if i < len(mask) and mask[i] else t)
    return "".join(out)


def compact_word_snippets_html(
    before_text: str,
    after_text: str,
    *,
    max_snips: Optional[int] = None,
) -> str:
    """
    Produces HTML with compact snippets, by diffing line-by-line and then token-highlighting
    within changed lines. Does not traverse across lines.
    """
    before_lines = (before_text or "").splitlines()
    after_lines = (after_text or "").splitlines()
    sm_lines = difflib.SequenceMatcher(None, before_lines, after_lines)

    dels: List[str] = []
    ins: List[str] = []
    truncated = False

    def push(kind: str, line_html: str) -> None:
        nonlocal truncated
        if truncated:
            return
        if max_snips is not None and (len(dels) + len(ins)) >= max_snips:
            truncated = True
            return
        (dels if kind == "del" else ins).append(line_html)

    for tag, i1, i2, j1, j2 in sm_lines.get_opcodes():
        if truncated:
            break
        if tag == "equal":
            continue

        # Pure adds/removes: highlight whole line
        if tag in ("delete", "insert"):
            lines = before_lines[i1:i2] if tag == "delete" else after_lines[j1:j2]
            cls = "diff-del" if tag == "delete" else "diff-ins"
            kind = "del" if tag == "delete" else "ins"
            for line in lines:
                toks = _tokenize(line)
                mask = [True] * len(toks)
                push(kind, _render_line_with_mask(toks, mask, cls))
                if truncated:
                    break
            continue

        # Replace blocks: compare line-by-line within the block
        old_block = before_lines[i1:i2]
        new_block = after_lines[j1:j2]
        n = max(len(old_block), len(new_block))

        for k in range(n):
            if truncated:
                break

            old_line = old_block[k] if k < len(old_block) else ""
            new_line = new_block[k] if k < len(new_block) else ""

            old_toks = _tokenize(old_line)
            new_toks = _tokenize(new_line)
            sm_tok = difflib.SequenceMatcher(None, old_toks, new_toks)

            old_mask = [False] * len(old_toks)
            new_mask = [False] * len(new_toks)

            for ttag, a1, a2, b1, b2 in sm_tok.get_opcodes():
                if ttag == "equal":
                    continue
                if ttag in ("delete", "replace"):
                    for i in range(a1, a2):
                        if 0 <= i < len(old_mask):
                            old_mask[i] = True
                if ttag in ("insert", "replace"):
                    for j in range(b1, b2):
                        if 0 <= j < len(new_mask):
                            new_mask[j] = True

            if any(old_mask) and old_toks:
                push("del", _render_line_with_mask(old_toks, old_mask, "diff-del"))
            if any(new_mask) and new_toks:
                push("ins", _render_line_with_mask(new_toks, new_mask, "diff-ins"))

    out_lines = [ln for ln in (dels + ins) if ln != ""]
    if truncated:
        out_lines.append("… (truncated)")
    if not out_lines:
        return ""
    return "<div class='diffbox'>" + "<br/>".join(out_lines) + "</div>"


def apply_edit_and_snapshot(
    recipe: Dict[str, Any],
    new_values: Dict[str, Any],
    label: str,
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], bool, Optional[str]]:
    """
    If changed: push snapshot of state-before-edit into versions and apply new values.
    Returns (updated_recipe, changed, version_id).
    """
    rr = ensure_recipe(recipe)
    before = {k: rr.get(k, "") for k in new_values.keys()}
    after = {k: new_values.get(k, "") for k in new_values.keys()}
    if before == after:
        return rr, False, None

    v = make_version_snapshot(rr, label=label, meta=meta)
    rr["versions"].insert(0, v)
    rr["versions"] = rr["versions"][:100]

    rr.update(new_values)
    rr["updated_at"] = now_iso()
    return rr, True, v.get("id")


def compute_version_diffs(recipe: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    version_id -> {saved_at,label,meta_lines,diff,associated_entry_id,associated_cook_on,diff_snip_html}

    Each version snapshot represents a "before" state. The diff shown is:
      snapshot(before) -> next state (next snapshot, or current recipe if newest).
    """
    rr = ensure_recipe(recipe)
    versions = rr.get("versions", []) or []
    diffs: Dict[str, Dict[str, Any]] = {}

    # Versions stored newest-first; compute diffs oldest->newest for stable before->after.
    snaps_oldest_first = list(reversed(versions))
    for i, before in enumerate(snaps_oldest_first):
        after = snaps_oldest_first[i + 1] if (i + 1) < len(snaps_oldest_first) else rr

        vid = before.get("id")
        if not vid:
            continue

        meta_lines: List[str] = []
        if (before.get("name", "") or "") != (after.get("name", "") or ""):
            meta_lines.append(f'Name: "{before.get("name","")}" → "{after.get("name","")}"')
        if (before.get("source", "") or "") != (after.get("source", "") or ""):
            meta_lines.append(f'Source: "{before.get("source","")}" → "{after.get("source","")}"')

        before_comb = combined_text(before)
        after_comb = combined_text(after)

        diffs[vid] = {
            "id": vid,
            "saved_at": before.get("saved_at", ""),
            "label": before.get("label", "Edit"),
            "meta_lines": meta_lines,
            "diff": changes_only(before_comb, after_comb, max_lines=None),
            "associated_entry_id": before.get("associated_entry_id", ""),
            "associated_cook_on": before.get("associated_cooked_on", ""),
            "diff_snip_html": compact_word_snippets_html(before_comb, after_comb, max_snips=None),
        }

    return diffs


# =============================================================================
# Timeline helpers
# =============================================================================
def entries_for_recipe(db: Dict[str, Any], rid: str) -> List[Dict[str, Any]]:
    return [ensure_entry(e) for e in db.get("entries", []) if e.get("recipe_id") == rid]


def cooks_for_recipe(db: Dict[str, Any], rid: str) -> List[Dict[str, Any]]:
    """Cook entries newest-first (anchored by cooked_on day)."""
    cooks = [e for e in entries_for_recipe(db, rid) if e.get("type") == "cook"]

    def cook_dt(e: Dict[str, Any]) -> datetime:
        day = _parse_date_only(e.get("cooked_on", "")) or _parse_date_only(e.get("created_at", "")) or date.min
        tod = _parse_time_only(e.get("created_at", ""))
        return datetime.combine(day, tod)

    cooks.sort(key=cook_dt, reverse=True)
    return cooks


def timeline_events_for_recipe(db: Dict[str, Any], rid: str) -> List[Dict[str, Any]]:
    """
    Stable-ordered events (oldest -> newest).
    Day anchoring:
      - cooks: cooked_on day
      - edits/notes/variation: created_at day
    Intra-day ordering uses created_at time-of-day, then original list order.
    """
    ev: List[Dict[str, Any]] = []
    entries = entries_for_recipe(db, rid)

    for idx, e in enumerate(entries):
        et = e.get("type")
        if et == "cook":
            day = _parse_date_only(e.get("cooked_on", "")) or _parse_date_only(e.get("created_at", ""))
        else:
            day = _parse_date_only(e.get("created_at", ""))
        tod = _parse_time_only(e.get("created_at", ""))
        ev.append({"dt": datetime.combine(day or date.min, tod), "entry": e, "_idx": idx})

    ev.sort(key=lambda x: (x["dt"], x["_idx"]))
    return ev


# =============================================================================
# Photos
# =============================================================================
def photo_dir_for(recipe_id: str) -> Path:
    ensure_dirs()
    p = PHOTOS_DIR / recipe_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_uploaded_photo(recipe_id: str, cook_entry_id: str, upload) -> Dict[str, Any]:
    """Persist a Streamlit UploadedFile under data/photos/<recipe_id>/."""
    original_name = (upload.name or "").strip() or "photo"

    ext = ""
    if "." in original_name:
        ext = "." + original_name.split(".")[-1].lower().strip()
        if len(ext) > 8:
            ext = ""

    pid = new_id()
    filename = f"{pid}{ext}"
    outpath = photo_dir_for(recipe_id) / filename
    outpath.write_bytes(upload.getbuffer())

    return {
        "id": pid,
        "recipe_id": recipe_id,
        "cook_entry_id": cook_entry_id,
        "filename": filename,
        "original_name": original_name,
        "created_at": now_iso(),
    }


def delete_photo_file(recipe_id: str, filename: str) -> None:
    try:
        p = photo_dir_for(recipe_id) / filename
        if p.exists():
            p.unlink()
    except Exception:
        pass


def photos_for_cook(db: Dict[str, Any], cook_entry_id: str) -> List[Dict[str, Any]]:
    items = [p for p in db.get("photos", []) if p.get("cook_entry_id") == cook_entry_id]
    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return items


# =============================================================================
# Migration / normalization
# =============================================================================
def normalize_db(db: Dict[str, Any]) -> bool:
    """
    Ensure required keys exist and migrate older shapes into the current schema.
    Returns True if db was modified.
    """
    dirty = False

    db.setdefault("version", SCHEMA_VERSION)
    db.setdefault("recipes", {})
    db.setdefault("entries", [])
    db.setdefault("photos", [])

    # ---- recipes ----
    for rid, r in list(db["recipes"].items()):
        rr = ensure_recipe(r)

        if not rr.get("original"):
            rr["original"] = make_original_snapshot(rr)
            dirty = True

        # Ensure version ids + required fields
        versions = rr.get("versions", []) or []
        for v in versions:
            if not v.get("id"):
                v["id"] = new_id()
                dirty = True
            v.setdefault("saved_at", rr.get("updated_at") or now_iso())
            v.setdefault("label", "Edit")
            for k in VERSION_FIELDS:
                v.setdefault(k, rr.get(k, ""))

        # Normalize variations:
        # - legacy list[str] => list[dict]
        # - legacy dict may use 'name' instead of 'title'
        raw_vars = rr.get("variations", []) or []
        norm_vars: List[Dict[str, Any]] = []

        for item in raw_vars:
            if isinstance(item, str):
                txt = item.rstrip()
                first = (txt.strip().splitlines() or ["Variation"])[0].strip()
                item = {"title": (first[:80] if first else "Variation"), "text": txt}
                dirty = True

            if isinstance(item, dict):
                if "title" not in item and "name" in item:
                    item["title"] = item.get("name") or ""
                    dirty = True

                if not (item.get("title") or "").strip():
                    txt = (item.get("text") or "").rstrip()
                    first = (txt.strip().splitlines() or ["Variation"])[0].strip()
                    item["title"] = (first[:80] if first else "Variation")
                    dirty = True

                norm_vars.append(ensure_variation(item))
            else:
                dirty = True

        rr["variations"] = norm_vars
        db["recipes"][rid] = rr

    # ---- migrate legacy cooks list -> entries ----
    if "cooks" in db and db.get("cooks") and not db.get("entries"):
        for c in db.get("cooks", []):
            e = {
                "id": c.get("id") or new_id(),
                "recipe_id": c.get("recipe_id", ""),
                "type": "cook",
                "created_at": c.get("created_at") or now_iso(),
                "cooked_on": c.get("cooked_on", ""),
                "cook_notes": c.get("notes", ""),
                "thoughts": "",
                "edited_recipe": bool(c.get("edited_recipe") or c.get("snapshotted") or c.get("associated_version_id")),
                "no_edit_reason": c.get("no_edit_reason", ""),
                "associated_version_id": c.get("associated_version_id", ""),
            }
            db["entries"].append(ensure_entry(e))
        dirty = True

    # ---- normalize entries ----
    db["entries"] = [ensure_entry(e) for e in db.get("entries", []) if isinstance(e, dict)]

    # ---- normalize photos ----
    norm_photos: List[Dict[str, Any]] = []
    for p in db.get("photos", []) or []:
        if not isinstance(p, dict):
            dirty = True
            continue
        p.setdefault("id", p.get("id") or new_id())
        p.setdefault("recipe_id", "")
        p.setdefault("cook_entry_id", "")
        p.setdefault("filename", "")
        p.setdefault("original_name", "")
        p.setdefault("created_at", p.get("created_at") or now_iso())
        norm_photos.append(p)
    db["photos"] = norm_photos

    return dirty


# =============================================================================
# UI helpers
# =============================================================================
def inject_css() -> None:
    # Deduplicated + kept class names stable (matches existing HTML rendering).
    st.markdown(
        """
        <style>
          .diffbox, pre.diffbox, .diff-inline-box, .note-pre {
            padding: 0.75rem 0.9rem;
            border-radius: 0.6rem;
            border: 1px solid rgba(120,120,120,0.25);
            background: rgba(120,120,120,0.08);
            white-space: pre-wrap;
            line-height: 1.35;
            font-size: 0.92rem;
            margin: 0.25rem 0 0.75rem 0;
          }

          .diffbox, pre.diffbox, .diff-inline-box {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
          }

          .diff-del { background: rgba(255, 0, 0, 0.18); }
          .diff-ins { background: rgba(0, 255, 0, 0.16); }

          .diff-del-inline { background: rgba(255, 0, 0, 0.18); text-decoration: line-through; }
          .diff-ins-inline { background: rgba(0, 255, 0, 0.16); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sorted_recipes(db_: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Recipes sorted newest-updated first, then name."""
    items: List[Tuple[str, Dict[str, Any]]] = []
    for rid, r in (db_.get("recipes", {}) or {}).items():
        items.append((rid, ensure_recipe(r)))
    items.sort(key=lambda x: (x[1].get("updated_at") or "", (x[1].get("name") or "")), reverse=True)
    return items


def build_recipe_labels(items: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, str]:
    """Map display label -> recipe_id (dedup labels by suffixing id prefix)."""
    labels: Dict[str, str] = {}
    used: Set[str] = set()
    for rid, r in items:
        base = (r.get("name") or "(untitled)").strip() or "(untitled)"
        lbl = base
        if lbl in used:
            lbl = f"{lbl} ({rid[:6]})"
        used.add(lbl)
        labels[lbl] = rid
    return labels


def render_note_pre(text: str) -> None:
    txt = (text or "").rstrip()
    if not txt:
        st.caption("(no text)")
        return
    st.markdown(f"<div class='note-pre'>{html.escape(txt)}</div>", unsafe_allow_html=True)


def render_variations_section(recipe: Dict[str, Any]) -> None:
    st.divider()
    st.markdown("### Variations")

    vars_ = _normalized_variations(recipe)
    if not vars_:
        st.caption("No variations saved yet.")
        return

    for v in vars_:
        with st.expander(variation_label(v, n=90), expanded=False):
            txt = (v.get("text") or "").strip()
            if txt:
                st.code(txt)
            else:
                st.caption("(no text)")


# =============================================================================
# UI: pages / tabs
# =============================================================================
def page_add_recipe(db: Dict[str, Any]) -> None:
    st.subheader("Add a recipe")

    with st.form("add_recipe", clear_on_submit=True):
        name = st.text_input("Name", placeholder="e.g., NYT Chocolate Chip Cookies")
        source = st.text_input("Source (optional)", placeholder="URL or site name")
        ingredients = st.text_area("Ingredients", height=200)
        instructions = st.text_area("Steps / method", height=240)
        submitted = st.form_submit_button("Save recipe")

    if not submitted:
        return
    if not name.strip():
        st.error("Name is required.")
        return

    rid = new_id()
    created = now_iso()
    recipe = ensure_recipe(
        {
            "id": rid,
            "name": name.strip(),
            "source": source.strip(),
            "ingredients": ingredients.rstrip(),
            "instructions": instructions.rstrip(),
            "created_at": created,
            "updated_at": created,
            "versions": [],
            "variations": [],
        }
    )
    recipe["original"] = make_original_snapshot(recipe)

    db["recipes"][rid] = recipe
    save_db(db)

    st.session_state["lib_focus_id"] = rid
    st.session_state["_page_request"] = "Library"
    st.success("Saved.")
    st.rerun()


def tab_original_recipe(recipe: Dict[str, Any]) -> None:
    original = recipe.get("original") or {}
    if not original:
        st.caption("No original snapshot found.")
        return

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("### Ingredients")
        if original.get("ingredients"):
            st.code(original["ingredients"])
        else:
            st.caption("No ingredients saved.")
    with c2:
        st.markdown("### Steps / method")
        if original.get("instructions"):
            st.write(original["instructions"])
        else:
            st.caption("No method saved.")


def tab_current_version(recipe: Dict[str, Any]) -> None:
    original = recipe.get("original") or {}
    show_diff = st.toggle(
        "Show changes vs original",
        value=False,
        key=f"cur_show_diff_{recipe.get('id','')}",
    )

    c1, c2 = st.columns(2, gap="large")

    # Original-vs-current tracked changes view
    if show_diff and original:
        name_changed = (original.get("name", "") or "") != (recipe.get("name", "") or "")
        source_changed = (original.get("source", "") or "") != (recipe.get("source", "") or "")
        if name_changed or source_changed:
            lines: List[str] = []
            if name_changed:
                lines.append(f'**Name:** "{original.get("name","")}" → "{recipe.get("name","")}"')
            if source_changed:
                lines.append(f'**Source:** "{original.get("source","")}" → "{recipe.get("source","")}"')
            st.caption(" · ".join(lines))

        orig_ing = original.get("ingredients", "") or ""
        cur_ing = recipe.get("ingredients", "") or ""
        orig_ins = original.get("instructions", "") or ""
        cur_ins = recipe.get("instructions", "") or ""

        with c1:
            st.markdown("### Ingredients (tracked changes vs original)")
            if orig_ing == cur_ing:
                if cur_ing.strip():
                    st.code(cur_ing)
                st.caption("No differences in ingredients vs original.")
            else:
                st.markdown(inline_tracked_changes_html(orig_ing, cur_ing), unsafe_allow_html=True)

        with c2:
            st.markdown("### Steps / method (tracked changes vs original)")
            if orig_ins == cur_ins:
                if cur_ins.strip():
                    st.write(cur_ins)
                st.caption("No differences in method vs original.")
            else:
                st.markdown(inline_tracked_changes_html(orig_ins, cur_ins), unsafe_allow_html=True)

        if (orig_ing == cur_ing) and (orig_ins == cur_ins) and not (name_changed or source_changed):
            st.success("Current version matches the original.")

        render_variations_section(recipe)
        return

    # Normal current view
    with c1:
        st.markdown("### Ingredients")
        if recipe.get("ingredients"):
            st.code(recipe["ingredients"])
        else:
            st.caption("No ingredients saved.")
    with c2:
        st.markdown("### Steps / method")
        if recipe.get("instructions"):
            st.write(recipe["instructions"])
        else:
            st.caption("No method saved.")

    if show_diff and not original:
        st.caption("No original snapshot found to diff against.")

    render_variations_section(recipe)


def tab_notebook(db: Dict[str, Any], rid: str, diffs_by_vid: Dict[str, Dict[str, Any]]) -> None:
    st.markdown("### Notebook")

    events = timeline_events_for_recipe(db, rid)
    if not events:
        st.caption("Nothing yet.")
        return

    def one_line(s: str, n: int = 240) -> str:
        s = " ".join((s or "").strip().splitlines()).strip()
        return s if len(s) <= n else (s[:n].rstrip() + "…")

    def tight_divider() -> None:
        st.markdown("<hr style='margin:0.35rem 0; opacity:0.25;'>", unsafe_allow_html=True)

    for ev in events:
        e = ev["entry"]
        et = e.get("type")

        # ---- Cook ----
        if et == "cook":
            cooked_txt = fmt_stamp(e.get("cooked_on") or e.get("created_at"))
            edited_tag = " (edited)" if e.get("edited_recipe") else ""

            thoughts = one_line(e.get("thoughts", ""))
            cook_notes = one_line(e.get("cook_notes", ""))

            vtitle = (e.get("variation_title") or "").strip()
            var_part = f"Variation: {vtitle}" if vtitle else ""

            summary_parts = [p for p in (var_part, thoughts, cook_notes) if p]
            summary = " — ".join(summary_parts)

            no_change_suffix = ""
            if not e.get("edited_recipe"):
                reason = (e.get("no_edit_reason") or "").strip()
                if reason:
                    no_change_suffix = f" — *No recipe change — {reason}*"

            if summary:
                st.markdown(f"**🍳 {cooked_txt}{edited_tag}:** {summary}{no_change_suffix}")
            else:
                st.markdown(f"**🍳 {cooked_txt}{edited_tag}**{no_change_suffix}")

            # If cook included recipe update, show ONLY the diff
            if e.get("edited_recipe"):
                vid = (e.get("associated_version_id") or "").strip()
                if vid and vid in diffs_by_vid:
                    vinfo = diffs_by_vid[vid]
                    snip = (vinfo.get("diff_snip_html") or "").strip()
                    diff_txt = (vinfo.get("diff") or "").strip()
                    if snip:
                        st.markdown(snip, unsafe_allow_html=True)
                    elif diff_txt:
                        st.markdown(f"```diff\n{diff_txt}\n```")
                    else:
                        st.caption("No ingredient/method line changes detected for this edit.")

            # Photos (compact)
            ph = photos_for_cook(db, e.get("id", ""))
            if ph:
                st.caption("Photos")
                cols = st.columns(min(4, len(ph)))
                for i, p in enumerate(ph[:8]):
                    img_path = photo_dir_for(rid) / (p.get("filename") or "")
                    with cols[i % len(cols)]:
                        if img_path.exists():
                            st.image(
                                str(img_path),
                                caption=fmt_stamp(p.get("created_at")),
                                width="stretch",
                            )

            tight_divider()
            continue

        # ---- Edit ----
        if et == "edit":
            dt_txt = fmt_stamp(e.get("created_at"))
            thoughts = (e.get("thoughts") or "").strip()
            if thoughts:
                st.markdown(f"**✏️ {dt_txt}:** {' '.join(thoughts.splitlines()).strip()}")
            else:
                st.markdown(f"**✏️ {dt_txt}**")

            vid = (e.get("associated_version_id") or "").strip()
            if vid and vid in diffs_by_vid:
                vinfo = diffs_by_vid[vid]
                snip = (vinfo.get("diff_snip_html") or "").strip()
                diff_txt = (vinfo.get("diff") or "").strip()
                if snip:
                    st.markdown(snip, unsafe_allow_html=True)
                elif diff_txt:
                    st.markdown(f"```diff\n{diff_txt}\n```")
                else:
                    st.caption("No ingredient/method line changes detected for this edit.")

            tight_divider()
            continue

        # ---- Variation event ----
        if et == "variation":
            dt_txt = fmt_stamp(e.get("created_at"))
            action = (e.get("variation_action") or "update").strip().lower()
            vtitle = (e.get("variation_title") or "").strip() or "(untitled variation)"
            vtext = (e.get("variation_text") or "").strip()

            verb = {"add": "Added", "edit": "Edited", "delete": "Deleted"}.get(action, "Updated")
            st.markdown(f"**🧪 {dt_txt}:** {verb} variation — {vtitle}")

            if vtext and action != "delete":
                render_note_pre(vtext)

            tight_divider()
            continue

        # ---- Note ----
        dt_txt = fmt_stamp(e.get("created_at"))
        note_txt = one_line(e.get("thoughts", ""))
        if note_txt:
            st.markdown(f"**📝 {dt_txt}:** {note_txt}")
        else:
            st.markdown(f"**📝 {dt_txt}**")
        tight_divider()


def tab_new_entry(db: Dict[str, Any], rid: str, recipe: Dict[str, Any]) -> None:
    st.markdown("### New entry")

    mode = st.radio(
        "What are you adding?",
        ["Log a cook", "Edit Recipe", "Notes only", "Add/Edit variations"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # -------------------------------------------------------------------------
    # Log a cook
    # -------------------------------------------------------------------------
    if mode == "Log a cook":
        st.caption("Log a cook + notes.")

        vars_ = _normalized_variations(recipe)

        used: Set[str] = set()
        var_label_to_id: Dict[str, str] = {}
        var_labels: List[str] = []

        for v in vars_:
            lbl = variation_label(v, n=60)
            if lbl in used:
                lbl = f"{lbl} ({v.get('id','')[:6]})"
            used.add(lbl)
            var_labels.append(lbl)
            var_label_to_id[lbl] = v.get("id", "")

        with st.form("new_cook", clear_on_submit=True):
            cooked_on = st.date_input("Cooked on", value=date.today())
            pick = st.selectbox(
                "Variation used (optional)",
                ["(none)"] + var_labels,
                index=0,
            )

            variation_id = var_label_to_id.get(pick, "") if pick != "(none)" else ""
            variation_title = ""
            variation_text = ""

            if variation_id:
                vv = next((x for x in vars_ if x.get("id") == variation_id), None)
                if vv:
                    variation_title = (vv.get("title") or "").strip()
                    variation_text = (vv.get("text") or "").strip()

            cook_notes = st.text_area(
                "Cook notes",
                height=220,
                placeholder=(
                    "Missing ingredients, workflow moments, oven rack location, timing, temps, substitutions, "
                    "results, ideas/goals for next time, etc. Also write fun memories about this day"
                ),
            )
            submit = st.form_submit_button("Save entry")

        if submit:
            entry_id = new_id()
            cook_entry = ensure_entry(
                {
                    "id": entry_id,
                    "recipe_id": rid,
                    "type": "cook",
                    "created_at": now_iso(),
                    "cooked_on": cooked_on.isoformat(),
                    "thoughts": "",
                    "cook_notes": cook_notes.strip(),
                    "edited_recipe": False,
                    "no_edit_reason": "",
                    "associated_version_id": "",
                    "variation_id": variation_id,
                    "variation_text": (variation_text or "").rstrip(),
                    "variation_title": variation_title,
                }
            )
            db["entries"].append(cook_entry)
            save_db(db)
            st.success("Saved.")
            st.rerun()

        return

    # -------------------------------------------------------------------------
    # Edit Recipe
    # -------------------------------------------------------------------------
    if mode == "Edit Recipe":
        st.caption("This creates a timeline entry + updates the recipe (tracked).")

        cooks = cooks_for_recipe(db, rid)
        last_cook = cooks[0] if cooks else None
        last_cook_lbl = fmt_stamp((last_cook or {}).get("cooked_on") or (last_cook or {}).get("created_at"))

        with st.form("new_edit", clear_on_submit=True):
            attach_to_last = False
            if last_cook:
                attach_to_last = st.checkbox(
                    f"Attach this edit to most recent cook ({last_cook_lbl})",
                    value=False,
                )
                if last_cook.get("edited_recipe"):
                    st.caption(
                        "Note: the most recent cook already has an attached edit; this will save as a standalone edit entry."
                    )
            else:
                st.caption("No cooks logged yet — this will save as a standalone edit entry.")

            thoughts = st.text_area(
                "General thoughts",
                height=140,
                placeholder="Why you’re changing it / what you want to improve…",
            )
            ingredients = st.text_area(
                "Ingredients",
                value=(recipe.get("ingredients") or ""),
                height=200,
            )
            instructions = st.text_area(
                "Steps / method",
                value=(recipe.get("instructions") or ""),
                height=240,
            )
            submit = st.form_submit_button("Save entry")

        if submit:
            attach_target = None
            if attach_to_last and last_cook and not last_cook.get("edited_recipe"):
                attach_target = last_cook

            entry_id = new_id()  # for standalone edit entries

            new_values = {
                "ingredients": ingredients.rstrip(),
                "instructions": instructions.rstrip(),
            }

            # Store association metadata inside the version snapshot
            meta: Dict[str, Any] = {"associated_entry_id": (attach_target.get("id") if attach_target else entry_id)}
            if attach_target:
                meta["associated_cooked_on"] = attach_target.get("cooked_on", "")

            label = f"Edit ({fmt_stamp(date.today())})"
            updated_recipe, changed, version_id = apply_edit_and_snapshot(
                db["recipes"][rid],
                new_values,
                label=label,
                meta=meta,
            )

            if not changed:
                st.info("No changes detected — entry not saved.")
                return

            # If attaching: mutate cook entry so Notebook shows diff within that cook log
            if attach_target:
                t = (thoughts or "").strip()
                if t:
                    prev = (attach_target.get("thoughts") or "").strip()
                    attach_target["thoughts"] = (prev + ("\n" if prev else "") + t).strip()

                attach_target["edited_recipe"] = True
                attach_target["associated_version_id"] = version_id or ""
                attach_target["no_edit_reason"] = ""

                db["recipes"][rid] = ensure_recipe(updated_recipe)
                save_db(db)
                st.success("Saved (attached to most recent cook).")
                st.rerun()
                return

            # Otherwise: standalone edit entry
            edit_entry = ensure_entry(
                {
                    "id": entry_id,
                    "recipe_id": rid,
                    "type": "edit",
                    "created_at": now_iso(),
                    "thoughts": (thoughts or "").strip(),
                    "associated_version_id": version_id or "",
                }
            )
            db["entries"].append(edit_entry)
            db["recipes"][rid] = ensure_recipe(updated_recipe)
            save_db(db)
            st.success("Saved.")
            st.rerun()

        return

    # -------------------------------------------------------------------------
    # Notes only
    # -------------------------------------------------------------------------
    if mode == "Notes only":
        st.caption("A pure note entry (no cook, no recipe change).")

        with st.form("new_note", clear_on_submit=True):
            thoughts = st.text_area("General thoughts / notes", height=200)
            submit = st.form_submit_button("Save entry")

        if submit:
            if not thoughts.strip():
                st.error("Please write something (or cancel).")
                return
            note_entry = ensure_entry(
                {
                    "id": new_id(),
                    "recipe_id": rid,
                    "type": "note",
                    "created_at": now_iso(),
                    "thoughts": thoughts.strip(),
                }
            )
            db["entries"].append(note_entry)
            save_db(db)
            st.success("Saved.")
            st.rerun()

        return

    # -------------------------------------------------------------------------
    # Add/Edit variations
    # -------------------------------------------------------------------------
    st.caption("Add a new variation, or edit/delete existing ones.")

    with st.form("new_variation", clear_on_submit=True):
        vtitle = st.text_input("Title", placeholder="e.g., Spicy weeknight version")
        vtext = st.text_area("Variation text", height=200, placeholder="Write the variation here…")
        submit_new = st.form_submit_button("Add variation")

    if submit_new:
        if not (vtitle or "").strip():
            st.error("Title is required.")
            return

        rr = ensure_recipe(db["recipes"][rid])
        rr["variations"] = _normalized_variations(rr)

        new_var = ensure_variation({"title": vtitle.strip(), "text": (vtext or "").rstrip()})
        rr["variations"].insert(0, new_var)
        rr["variations"] = rr["variations"][:100]
        rr["updated_at"] = now_iso()
        db["recipes"][rid] = rr

        db["entries"].append(
            ensure_entry(
                {
                    "id": new_id(),
                    "recipe_id": rid,
                    "type": "variation",
                    "created_at": now_iso(),
                    "thoughts": f"Added variation: {(new_var.get('title') or '').strip()}",
                    "variation_action": "add",
                    "variation_id": new_var.get("id", ""),
                    "variation_title": (new_var.get("title") or "").strip(),
                    "variation_text": (new_var.get("text") or "").rstrip(),
                }
            )
        )

        save_db(db)
        st.success("Saved.")
        st.rerun()

    st.divider()
    st.markdown("### Existing variations")

    rr = ensure_recipe(db["recipes"][rid])
    rr["variations"] = _normalized_variations(rr)
    vars_ = rr["variations"]

    if not vars_:
        st.caption("No variations yet.")
        return

    for v in vars_:
        vid = v.get("id", "")
        key_title = f"var_title_{rid}_{vid}"
        key_text = f"var_text_{rid}_{vid}"

        with st.expander(variation_label(v, n=90), expanded=False):
            st.text_input("Title", value=v.get("title", ""), key=key_title)
            st.text_area("Variation text", value=v.get("text", ""), height=200, key=key_text)

            col1, col2 = st.columns([0.25, 0.25], gap="small")

            if col1.button("Save changes", key=f"var_save_{rid}_{vid}"):
                new_title = (st.session_state.get(key_title, "") or "").strip()
                new_text = (st.session_state.get(key_text, "") or "").rstrip()
                if not new_title:
                    st.error("Title is required.")
                    return

                for vv in rr.get("variations", []) or []:
                    if isinstance(vv, dict) and vv.get("id") == vid:
                        vv["title"] = new_title
                        vv["text"] = new_text
                        vv["updated_at"] = now_iso()
                        break

                rr["updated_at"] = now_iso()
                db["recipes"][rid] = ensure_recipe(rr)

                db["entries"].append(
                    ensure_entry(
                        {
                            "id": new_id(),
                            "recipe_id": rid,
                            "type": "variation",
                            "created_at": now_iso(),
                            "thoughts": f"Edited variation: {new_title}",
                            "variation_action": "edit",
                            "variation_id": vid,
                            "variation_title": new_title,
                            "variation_text": new_text,
                        }
                    )
                )

                save_db(db)
                st.success("Updated.")
                st.rerun()

            if col2.button("Delete", key=f"var_del_{rid}_{vid}"):
                old = next(
                    (x for x in (rr.get("variations", []) or []) if isinstance(x, dict) and x.get("id") == vid),
                    None,
                )
                old_title = (old.get("title") if old else "") or "(untitled variation)"
                old_text = (old.get("text") if old else "") or ""

                db["entries"].append(
                    ensure_entry(
                        {
                            "id": new_id(),
                            "recipe_id": rid,
                            "type": "variation",
                            "created_at": now_iso(),
                            "thoughts": f"Deleted variation: {old_title}",
                            "variation_action": "delete",
                            "variation_id": vid,
                            "variation_title": (old_title or "").strip(),
                            "variation_text": (old_text or "").rstrip(),
                        }
                    )
                )

                rr["variations"] = [
                    x for x in (rr.get("variations", []) or [])
                    if not (isinstance(x, dict) and x.get("id") == vid)
                ]
                rr["updated_at"] = now_iso()
                db["recipes"][rid] = ensure_recipe(rr)
                save_db(db)
                st.success("Deleted.")
                st.rerun()


def tab_photos(db: Dict[str, Any], rid: str) -> None:
    st.markdown("### Photos (associated with a cook)")

    cooks = cooks_for_recipe(db, rid)
    if not cooks:
        st.info("Log a cook first — photos must be attached to a cook.")
        return

    cook_labels: List[str] = []
    cook_map: Dict[str, str] = {}
    for c in cooks:
        lbl = f"{fmt_stamp(c.get('cooked_on'))}"
        tail = (c.get("cook_notes") or "").strip().replace("\n", " ")
        if tail:
            lbl = f"{lbl} — {tail[:45]}{'…' if len(tail) > 45 else ''}"
        if lbl in cook_map:
            lbl = f"{lbl} ({c.get('id','')[:6]})"
        cook_labels.append(lbl)
        cook_map[lbl] = c.get("id")

    pick_lbl = st.selectbox("Attach photos to cook", cook_labels, index=0)
    cook_id = cook_map[pick_lbl]

    # Prevent duplicate uploads on rerun: require explicit submit + reset uploader key
    nonce_key = f"photo_uploader_nonce_{rid}"
    st.session_state.setdefault(nonce_key, 0)
    uploader_key = f"photo_uploader_{rid}_{cook_id}_{st.session_state[nonce_key]}"

    with st.form(f"upload_photos_form_{rid}_{cook_id}", clear_on_submit=False):
        uploads = st.file_uploader(
            "Add photos",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=uploader_key,
        )
        do_upload = st.form_submit_button("Upload selected")

    if do_upload:
        if not uploads:
            st.info("Pick one or more files first.")
        else:
            added = 0
            for up in uploads:
                db["photos"].append(save_uploaded_photo(rid, cook_id, up))
                added += 1
            save_db(db)

            # bump nonce so the uploader resets on rerun
            st.session_state[nonce_key] += 1

            st.success(f"Added {added} photo(s).")
            st.rerun()

    st.divider()
    st.markdown("### Photos for this cook")

    ph = photos_for_cook(db, cook_id)
    if not ph:
        st.caption("No photos attached to this cook yet.")
        return

    for p in ph:
        img_path = photo_dir_for(rid) / (p.get("filename") or "")
        rowL, rowR = st.columns([0.78, 0.22])
        with rowL:
            if img_path.exists():
                st.image(
                    str(img_path),
                    caption=f"{fmt_stamp(p.get('created_at'))} · {p.get('original_name','')}",
                    width="stretch",
                )
            else:
                st.warning(f"Missing file: {p.get('filename','')}")
        with rowR:
            if st.button("Delete", key=f"del_photo_{p.get('id')}", width="stretch"):
                delete_photo_file(rid, p.get("filename", ""))
                db["photos"] = [x for x in db.get("photos", []) if x.get("id") != p.get("id")]
                save_db(db)
                st.success("Deleted.")
                st.rerun()


def page_calendar(db: Dict[str, Any]) -> None:
    st.subheader("Calendar")
    st.caption("A chronological list of cook logs across all recipes.")

    c1, c2, c3 = st.columns([0.34, 0.33, 0.33], gap="small")
    with c1:
        show_comments = st.toggle("Show comments", value=True, key="cal_show_comments")
    with c2:
        show_photos = st.toggle("Show photos", value=False, key="cal_show_photos")
    with c3:
        newest_first = st.toggle("Newest first", value=True, key="cal_newest_first")

    cooks: List[Tuple[datetime, Dict[str, Any], str, Dict[str, Any]]] = []
    for raw in (db.get("entries", []) or []):
        if not isinstance(raw, dict):
            continue
        e = ensure_entry(raw)
        if e.get("type") != "cook":
            continue

        rid = (e.get("recipe_id") or "").strip()
        r_raw = (db.get("recipes", {}) or {}).get(rid) or {"id": rid, "name": "(missing recipe)"}
        r = ensure_recipe(dict(r_raw))

        day = _parse_date_only(e.get("cooked_on", "")) or _parse_date_only(e.get("created_at", ""))
        tod = _parse_time_only(e.get("created_at", ""))
        dt = datetime.combine(day or date.min, tod)

        cooks.append((dt, e, rid, r))

    if not cooks:
        st.info("No cooks logged yet.")
        return

    cooks.sort(key=lambda x: x[0], reverse=newest_first)

    current_day: Optional[date] = None
    first_group = True

    for dt, e, rid, r in cooks:
        d = dt.date()
        if current_day != d:
            if not first_group:
                st.divider()
            first_group = False
            current_day = d
            st.markdown(f"### {fmt_stamp(d)}")

        recipe_name = (r.get("name") or "(untitled)").strip() or "(untitled)"
        edited_tag = " · **edited**" if e.get("edited_recipe") else ""
        var_title = (e.get("variation_title") or "").strip()
        var_tag = f" · 🧪 {var_title}" if var_title else ""

        st.markdown(f"**🍳 {recipe_name}**{var_tag}{edited_tag}")

        if show_comments:
            thoughts = (e.get("thoughts") or "").rstrip()
            cook_notes = (e.get("cook_notes") or "").rstrip()

            if thoughts.strip():
                st.caption("Thoughts")
                render_note_pre(thoughts)
            if cook_notes.strip():
                render_note_pre(cook_notes)
            if not (thoughts.strip() or cook_notes.strip()):
                st.caption("No comments for this cook.")

        if show_photos:
            ph = photos_for_cook(db, e.get("id", ""))
            if not ph:
                st.caption("No photos for this cook.")
            else:
                st.caption("Photos")
                cols = st.columns(min(4, len(ph)), gap="small")
                for i, p in enumerate(ph[:8]):
                    img_path = photo_dir_for(rid) / (p.get("filename") or "")
                    with cols[i % len(cols)]:
                        if img_path.exists():
                            st.image(
                                str(img_path),
                                caption=fmt_stamp(p.get("created_at")),
                                width="stretch",
                            )


def page_library(db: Dict[str, Any]) -> None:
    items = sorted_recipes(db)
    if not items:
        st.info("No recipes yet. Go to **Add recipe** in the sidebar.")
        st.stop()

    labels = build_recipe_labels(items)
    labels_list = list(labels.keys())
    ids_list = [labels[l] for l in labels_list]

    focus_id = st.session_state.pop("lib_focus_id", None)
    default_id = focus_id or st.session_state.get("lib_current_id") or ids_list[0]
    default_index = ids_list.index(default_id) if default_id in ids_list else 0

    left, right = st.columns([0.48, 0.52], gap="large")
    with left:
        chosen_label = st.selectbox("Select a recipe", labels_list, index=default_index, key="lib_select")
        rid = labels[chosen_label]
        st.session_state["lib_current_id"] = rid

    recipe = ensure_recipe(db["recipes"][rid])
    db["recipes"][rid] = recipe
    diffs_by_vid = compute_version_diffs(recipe)

    with right:
        st.markdown(f"## {(recipe.get('name') or '(untitled)').strip()}")
        bits: List[str] = []
        if recipe.get("source"):
            bits.append(f"Source: {recipe['source']}")
        bits.append(f"Updated: {fmt_stamp(recipe.get('updated_at'))}")
        st.caption(" · ".join([b for b in bits if b]))

    st.divider()

    tab_original, tab_timeline, tab_current, tab_new_entry_, tab_photos_ = st.tabs(
        ["Original recipe", "Notebook", "Current version", "New entry", "Photos"]
    )

    with tab_original:
        tab_original_recipe(recipe)

    with tab_timeline:
        tab_notebook(db, rid, diffs_by_vid)

    with tab_current:
        tab_current_version(recipe)

    with tab_new_entry_:
        tab_new_entry(db, rid, recipe)

    with tab_photos_:
        tab_photos(db, rid)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    st.set_page_config(page_title="Recipe Log", page_icon="🍳", layout="wide")
    inject_css()

    # Safe navigation: avoid setting st.session_state["page"] after radio is created
    if "_page_request" in st.session_state:
        st.session_state["page"] = st.session_state.pop("_page_request")

    db = load_db()
    if normalize_db(db):
        save_db(db)

    st.title("🍳 Recipe Log")
    st.caption("Save a recipe → log cooks & thoughts → optionally update recipe → review everything chronologically.")

    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Go to",
            ["Library", "Add recipe", "Calendar"],
            index=0,
            key="page",
            label_visibility="collapsed",
        )
        st.divider()
        st.subheader("Quick stats")
        st.write(f"Recipes: **{len(db.get('recipes', {}))}**")

    if page == "Add recipe":
        page_add_recipe(db)
    elif page == "Calendar":
        page_calendar(db)
    else:
        page_library(db)


main()
