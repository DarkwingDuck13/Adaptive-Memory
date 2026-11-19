#!/usr/bin/env python3
"""Read-only memory health report for the Adaptive Memory filter's Open WebUI memories.

Written by Claude (Anthropic) together with DarkwingDuck13. MIT License.
Updated 2026-11-18 (also written by Claude): lines that start with the user's name count as
facts about the user, new flags (raw quotes, speaker labels, "you" in a fact about the user,
transient scene state, health details, long lines), a looser near-duplicate check, and
`--split` to compare lines before and after a change such as a model switch.

What it does
  Opens Open WebUI's database READ-ONLY, sorts every memory into a kind, prints counts,
  and writes ONE export file holding only the facts about the user. It never changes
  anything, and it never prints or exports a persona's own writing.

Kinds
  hers     content starts with a persona name ("Alice: ..."), or is tagged reflection
           or diary, or starts with "(your own note)". Only counted, never shown.
  user     starts with "User" or with the user's name ("Sam ...")
  story    starts with "In the story,"
  other    anything else

Flags (reasons a line deserves a look; none of them is a verdict)
  bad_start            kind "other" that is not one of the specific cases below
  speaker_label        starts with a speaker label ("Sam:", "User:", "Alice:") = a raw quote
  asterisk_action      contains *action text*, so it is roleplay text copied in
  second_person_start  starts with "You" or "Your" (the memory is addressed to someone)
  first_person_start   starts with "I", "I'm", "We", "Yes", "Maybe" (a quote, not a fact)
  you_in_user_fact     a fact about the user that says "you" or "your": which persona is
                       "you"? Facts about the user are shared by both personas.
  transient            "currently" or "right now": a scene state, not a lasting fact
  health               mentions a diagnosis or condition (your call whether it stays)
  long                 over 300 characters
  summary_leak         starts like a model reply ("Here is a summary ...")
  example_word:<w>     contains a word from the extractor prompt's examples (John, Jane,
                       Orlando, 32810, Holmes County, Best Buy, Kubernetes, Rust)
  too_short            under 20 characters of content
  low_confidence       confidence under 0.50
  near_dup:<id>        80% or more similar to an earlier line (the id is the earlier one)

Usage
  python3 memory_review.py                                  # report + export file
  python3 memory_review.py --db /path/to/webui.db           # another database path
  python3 memory_review.py --flags-only                     # also print only flagged lines
  python3 memory_review.py --split "2026-11-17 12:00"       # compare before and after
  python3 memory_review.py --out my_export.txt              # export file name
  python3 memory_review.py --persona Alice --persona Bob   # persona names (default)
  python3 memory_review.py --user-name Sam              # the user's name (default)

Standard library only. Safe to run while Open WebUI is running.
"""
import argparse
import difflib
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime

DEFAULT_DB = "/root/.open-webui/webui.db"
EXAMPLE_WORDS = ("John", "Jane", "Orlando", "32810", "Holmes County", "Best Buy", "Kubernetes", "Rust")
HER_TAGS = {"reflection", "diary"}
STORAGE = re.compile(
    r"^\[Tags:\s*(?P<tags>[^\]]*)\]\s*(?P<content>.*?)\s*\[Memory Bank:\s*(?P<bank>[^\]]+)\]\s*"
    r"\[Confidence:\s*(?P<conf>[^\]]+)\]",
    re.DOTALL,
)
LABEL = re.compile(r"^(User|Assistant|AI|Human)\s*:", re.IGNORECASE)
HEALTH = re.compile(r"autis|adhd|depress|anxiety|anxious|neurodiver|diagnos|disorder", re.IGNORECASE)


def parse(raw):
    text = (raw or "").strip()
    m = STORAGE.match(text)
    if not m:
        return {"content": text, "tags": [], "bank": "?", "conf": None}
    tags = [t.strip().lower() for t in m.group("tags").split(",") if t.strip() and t.strip().lower() != "none"]
    try:
        conf = float(m.group("conf").strip())
    except ValueError:
        conf = None
    return {"content": m.group("content").strip(), "tags": tags, "bank": m.group("bank").strip() or "?", "conf": conf}


def classify(rec, personas, user_name):
    c = rec["content"]
    low = c.lower()
    for name in personas:
        if re.match(rf"^{re.escape(name)}\s*[:,\-]", c, flags=re.IGNORECASE):
            return "hers", name
    if HER_TAGS & set(rec["tags"]) or low.startswith("(your own note)"):
        return "hers", "?"
    if re.match(rf"^(user|{re.escape(user_name)})\b", c, flags=re.IGNORECASE):
        return "user", ""
    if low.startswith("in the story,"):
        return "story", ""
    return "other", ""


def flags_for(rec, kind):
    c = rec["content"]
    flags = []
    if LABEL.match(c):
        flags.append("speaker_label")
    if len(re.findall(r"\*[^*]{3,}\*", c)) >= 1:
        flags.append("asterisk_action")
    if re.match(r"^(You|Your)\b", c):
        flags.append("second_person_start")
    if re.match(r"^(I|I'm|I've|I'll|We|Yes|Maybe)\b", c):
        flags.append("first_person_start")
    if kind == "user" and re.search(r"\b(you|your|you're|you've|yours)\b", c, flags=re.IGNORECASE):
        flags.append("you_in_user_fact")
    if re.search(r"\b(currently|right now)\b", c, flags=re.IGNORECASE):
        flags.append("transient")
    if HEALTH.search(c):
        flags.append("health")
    if len(c) > 300:
        flags.append("long")
    if re.match(r"^(Here is|Here's|Sure|Certainly)\b", c):
        flags.append("summary_leak")
    if kind == "other" and not flags:
        flags.append("bad_start")
    for w in EXAMPLE_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", c):
            flags.append(f"example_word:{w}")
    if len(c) < 20:
        flags.append("too_short")
    if rec["conf"] is not None and rec["conf"] < 0.5:
        flags.append("low_confidence")
    return flags


def when(ts):
    if ts is None:
        return None
    ts = float(ts)
    while ts > 1e11:  # milliseconds or nanoseconds
        ts /= 1000.0
    return datetime.fromtimestamp(ts)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Read-only memory health report")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--out", default="memory_review_export.txt")
    ap.add_argument("--persona", action="append", help="persona name (repeat for each); default Assistant")
    ap.add_argument("--user-name", default="User", help="the user's name; lines starting with it count as facts about the user")
    ap.add_argument("--dup", type=float, default=0.80, help="similarity for the near-duplicate flag (default 0.80)")
    ap.add_argument("--split", default="", help='"YYYY-MM-DD HH:MM": also compare flag counts before and after this time')
    ap.add_argument("--flags-only", action="store_true", help="also print only the flagged lines")
    args = ap.parse_args(argv)
    personas = args.persona or ["Assistant"]
    split = datetime.strptime(args.split, "%Y-%m-%d %H:%M") if args.split else None

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    rows = con.execute("select id, user_id, content, created_at, type from memory order by created_at").fetchall()
    con.close()

    kinds = Counter()
    hers_by = Counter()
    per_day = Counter()
    banks = Counter()
    flag_counts = Counter()
    side_flags = {"before": Counter(), "after": Counter()}
    side_n = Counter()
    exported = []
    seen = []  # (id, content) of earlier user/story/other lines, for near-duplicate checks

    for mid, uid, raw, created, mtype in rows:
        rec = parse(raw)
        kind, who = classify(rec, personas, args.user_name)
        kinds[kind] += 1
        dt = when(created)
        if kind == "hers":
            hers_by[who] += 1  # counted only: never shown or exported
            continue
        banks[rec["bank"]] += 1
        if dt:
            per_day[dt.strftime("%Y-%m-%d")] += 1
        flags = flags_for(rec, kind)
        for other_id, other_text in seen[-400:]:
            if difflib.SequenceMatcher(None, rec["content"].lower(), other_text.lower()).ratio() >= args.dup:
                flags.append(f"near_dup:{str(other_id)[:8]}")
                break
        seen.append((mid, rec["content"]))
        for f in flags:
            flag_counts[f.split(":")[0]] += 1
        if split and dt:
            side = "after" if dt >= split else "before"
            side_n[side] += 1
            if flags:
                side_flags[side]["any_flag"] += 1
            for f in flags:
                side_flags[side][f.split(":")[0]] += 1
        exported.append((str(mid)[:8], dt.strftime("%Y-%m-%d %H:%M") if dt else "?", rec["bank"],
                         "?" if rec["conf"] is None else f"{rec['conf']:.2f}", kind, flags, rec["content"]))

    print(f"Database: {args.db} (opened read-only)")
    print(f"Memories: {len(rows)} total")
    for k in ("user", "story", "other", "hers"):
        print(f"  {k:6} {kinds[k]}")
    if hers_by:
        print("  hers, by persona (counts only, never shown): " + ", ".join(f"{k}={v}" for k, v in sorted(hers_by.items())))
    print("Flagged lines by reason: " + (", ".join(f"{k}={v}" for k, v in sorted(flag_counts.items())) or "none"))
    flagged = sum(1 for e in exported if e[5])
    print(f"Lines with at least one flag: {flagged} of {len(exported)}")
    print("Created per day (user, story, other): " + ", ".join(f"{d}={n}" for d, n in sorted(per_day.items())[-10:]))
    print("Memory banks: " + ", ".join(f"{b}={n}" for b, n in banks.most_common()))

    if split:
        print(f"\nBefore {args.split}: {side_n['before']} lines. After: {side_n['after']} lines.")
        reasons = sorted(set(side_flags['before']) | set(side_flags['after']))
        for r in reasons:
            print(f"  {r:22} before={side_flags['before'][r]:3}  after={side_flags['after'][r]:3}")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# Facts about the user only. A persona's own writing is never included.\n")
        f.write("# id | created | bank | confidence | kind | flags | content\n")
        for mid, d, bank, conf, kind, flags, content in exported:
            f.write(f"{mid} | {d} | {bank} | {conf} | {kind} | {','.join(flags) or '-'} | {content}\n")
    print(f"Wrote {len(exported)} lines to {args.out}")

    if args.flags_only:
        print("\nFlagged lines:")
        for mid, d, bank, conf, kind, flags, content in exported:
            if flags:
                print(f"{mid} | {d} | {','.join(flags)} | {content}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
