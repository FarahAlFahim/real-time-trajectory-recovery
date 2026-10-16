#!/usr/bin/env python3
"""Apply a Codex `*** Begin Patch` envelope to the working tree (fuzzy, best-effort).

Used as a SALVAGE step by src/runners/codex_swebench_lite.py: gpt-5-mini often emits a correct
apply_patch block but applies it with the wrong tool (e.g. `git apply`, which can't parse the
`*** Update File:` format) or a no-op script, leaving an empty diff. We extract the block from
the trajectory and apply it ourselves with this parser.

Reads the patch text from stdin (or argv[1]); applies relative to CWD. Prints what it changed.
Exit 0 if at least one file changed, 2 if nothing matched/changed, 1 on malformed input.

Supports: *** Update File / *** Add File / *** Delete File / *** Move to. Update hunks use the
apply_patch line prefixes (' ' context, '-' remove, '+' add) with optional `@@` section markers.
Matching is fuzzy: exact -> rstrip -> strip, searched from the last applied position then globally.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _extract_envelope(text: str) -> str:
    begin = text.find("*** Begin Patch")
    end = text.find("*** End Patch")
    if begin != -1 and end != -1 and end > begin:
        return text[begin + len("*** Begin Patch"): end]
    # Tolerate a missing Begin/End wrapper if the section markers are present.
    if "*** Update File:" in text or "*** Add File:" in text or "*** Delete File:" in text:
        return text
    return ""


def _fuzzy_find(hay: list[str], needle: list[str], start: int) -> int:
    """Index in `hay` where `needle` matches, trying exact then whitespace-insensitive."""
    if not needle:
        return start if start <= len(hay) else -1
    n = len(needle)
    for norm in (lambda s: s, lambda s: s.rstrip(), lambda s: s.strip()):
        nd = [norm(x) for x in needle]
        for base in (start, 0):
            for i in range(base, len(hay) - n + 1):
                if [norm(hay[i + j]) for j in range(n)] == nd:
                    return i
    return -1


def _apply_update(path: Path, hunk_lines: list[str]) -> bool:
    original = path.read_text()
    lines = original.split("\n")
    # Split the body into hunks on `@@` markers.
    hunks: list[list[str]] = []
    current: list[str] = []
    for ln in hunk_lines:
        if ln.startswith("@@"):
            if current:
                hunks.append(current)
                current = []
            continue
        current.append(ln)
    if current:
        hunks.append(current)

    pos = 0
    changed = False
    for hunk in hunks:
        before: list[str] = []
        after: list[str] = []
        for ln in hunk:
            # A truly empty line (no prefix) is an artifact of splitting/spacing, not hunk
            # content — a real blank context line carries a leading space. Skip it so it does
            # not corrupt the context block being matched.
            if ln == "":
                continue
            tag, text = (ln[:1], ln[1:]) if ln[:1] in (" ", "-", "+") else (" ", ln)
            if tag == " ":
                before.append(text)
                after.append(text)
            elif tag == "-":
                before.append(text)
            elif tag == "+":
                after.append(text)
        if before == after:
            continue
        idx = _fuzzy_find(lines, before, pos) if before else pos
        if idx >= 0:
            lines[idx:idx + len(before)] = after
            pos = idx + len(after)
            changed = True
            continue
        # Lenient fallback: the model frequently emits hunks with hallucinated/incomplete
        # context (e.g. two context lines that aren't actually adjacent), so the full block
        # can't match. Since we only need a NON-EMPTY, best-effort delivery (correctness is
        # Codex's job), anchor on a single real context/deleted line and splice the change in.
        if _apply_hunk_lenient(lines, hunk):
            changed = True
    if changed:
        path.write_text("\n".join(lines))
    return changed


def _apply_hunk_lenient(lines: list[str], hunk: list[str]) -> bool:
    """Best-effort: delete any matching '-' lines, insert '+' lines after a real context anchor."""
    ops: list[tuple[str, str]] = []
    for ln in hunk:
        if ln == "":
            continue
        tag = ln[:1] if ln[:1] in (" ", "-", "+") else " "
        text = ln[1:] if ln[:1] in (" ", "-", "+") else ln
        ops.append((tag, text))
    additions = [t for g, t in ops if g == "+"]
    deletions = [t for g, t in ops if g == "-"]
    changed = False
    # Remove deleted lines individually (fuzzy single-line).
    for dl in deletions:
        i = _fuzzy_find(lines, [dl], 0)
        if i >= 0:
            del lines[i]
            changed = True
    if additions:
        # Anchor = the context line immediately before the first addition, else any context
        # line, else the nearest deletion target. Insert additions right after it.
        ctx_before = None
        for g, t in ops:
            if g == "+":
                break
            if g == " ":
                ctx_before = t
        anchor = _fuzzy_find(lines, [ctx_before], 0) if ctx_before is not None else -1
        if anchor < 0:
            for g, t in ops:
                if g == " ":
                    anchor = _fuzzy_find(lines, [t], 0)
                    if anchor >= 0:
                        break
        if anchor >= 0:
            lines[anchor + 1:anchor + 1] = additions
            changed = True
    return changed


def _apply(text: str) -> int:
    body = _extract_envelope(text)
    if not body:
        print("apply_patch_salvage: no patch envelope found", file=sys.stderr)
        return 1
    lines = body.split("\n")
    changed_any = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("*** Add File:"):
            path = Path(line.split(":", 1)[1].strip())
            i += 1
            content: list[str] = []
            while i < len(lines) and not lines[i].startswith("*** "):
                content.append(lines[i][1:] if lines[i].startswith("+") else lines[i])
                i += 1
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(content))
            changed_any = True
            continue
        if line.startswith("*** Delete File:"):
            path = Path(line.split(":", 1)[1].strip())
            if path.exists():
                path.unlink()
                changed_any = True
            i += 1
            continue
        if line.startswith("*** Update File:"):
            path = Path(line.split(":", 1)[1].strip())
            i += 1
            # Optional "*** Move to:" line.
            move_to = None
            if i < len(lines) and lines[i].startswith("*** Move to:"):
                move_to = Path(lines[i].split(":", 1)[1].strip())
                i += 1
            hunk_lines: list[str] = []
            while i < len(lines) and not lines[i].startswith("*** "):
                hunk_lines.append(lines[i])
                i += 1
            if not path.exists():
                print(f"apply_patch_salvage: target missing: {path}", file=sys.stderr)
                continue
            if _apply_update(path, hunk_lines):
                changed_any = True
            if move_to is not None:
                move_to.parent.mkdir(parents=True, exist_ok=True)
                path.rename(move_to)
                changed_any = True
            continue
        i += 1
    if changed_any:
        print("apply_patch_salvage: applied")
        return 0
    print("apply_patch_salvage: nothing changed", file=sys.stderr)
    return 2


def main() -> int:
    text = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
    return _apply(text)


if __name__ == "__main__":
    raise SystemExit(main())
