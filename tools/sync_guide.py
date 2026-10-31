#!/usr/bin/env python3
"""Keeps the code blocks in docs/guide identical to the regions of tests/GuideSamples that CI builds and runs.

A block is marked in the Markdown by a comment line `<!-- sample: region-name -->` directly followed by a fenced csharp block.
The block's body is the `#region region-name` ... `#endregion` of a .cs file under tests/GuideSamples, dedented.

    python3 tools/sync_guide.py          rewrite the blocks
    python3 tools/sync_guide.py --check  fail (exit 1) if any block is out of date, missing its region, or unmarked
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "tests" / "GuideSamples"
GUIDE = ROOT / "docs" / "guide"


def regions():
    found = {}
    for path in sorted(SAMPLES.glob("*.cs")):
        lines = path.read_text(encoding="utf-8").split("\n")
        i = 0
        while i < len(lines):
            m = re.match(r"\s*#region\s+(\S+)\s*$", lines[i])
            if m:
                name = m.group(1)
                depth, body = 1, []
                i += 1
                while i < len(lines):
                    if re.match(r"\s*#region\b", lines[i]):
                        depth += 1
                    elif re.match(r"\s*#endregion\b", lines[i]):
                        depth -= 1
                        if depth == 0:
                            break
                    body.append(lines[i])
                    i += 1
                if name in found:
                    sys.exit(f"duplicate region {name}")
                found[name] = dedent(body)
            i += 1
    return found


def dedent(body):
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    indent = min((len(l) - len(l.lstrip()) for l in body if l.strip()), default=0)
    return "\n".join(l[indent:] if l.strip() else "" for l in body)


def main():
    check = "--check" in sys.argv
    code = regions()
    bad = []
    used = set()
    for md in sorted(GUIDE.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        out = []
        lines = text.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]
            out.append(line)
            m = re.match(r"<!-- sample: (\S+) -->$", line.strip())
            if m:
                name = m.group(1)
                if name not in code:
                    bad.append(f"{md.name}: no region '{name}'")
                    i += 1
                    continue
                used.add(name)
                if i + 1 >= len(lines) or not lines[i + 1].startswith("```csharp"):
                    bad.append(f"{md.name}: marker '{name}' is not followed by a csharp fence")
                    i += 1
                    continue
                j = i + 2
                while j < len(lines) and not lines[j].startswith("```"):
                    j += 1
                out.append("```csharp")
                out.extend(code[name].split("\n"))
                out.append("```")
                i = j + 1
                continue
            if line.startswith("```csharp") and not (i > 0 and lines[i - 1].strip().startswith("<!-- sample:")):
                bad.append(f"{md.name}:{i + 1}: a csharp block without a sample marker (use ```text for code that is not built)")
            i += 1
        new = "\n".join(out)
        if new != text:
            if check:
                bad.append(f"{md.name}: out of date (run tools/sync_guide.py)")
            else:
                md.write_text(new, encoding="utf-8")
    unused = sorted(set(code) - used)
    if unused and check:
        bad.append("regions never shown in the guide: " + ", ".join(unused))
    if bad:
        print("\n".join(bad))
        sys.exit(1)


main()
