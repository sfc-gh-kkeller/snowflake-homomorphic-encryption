"""Print the text outputs of executed notebooks (stream, results, errors) for quick review.

    pixi run python scripts/show_outputs.py notebooks/01_basics.ipynb [...]
"""

import html
import json
import re
import sys


def table_rows(h, limit=12):
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", h, re.S)[:limit]:
        cells = [html.unescape(re.sub("<.*?>", "", c)).strip() for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, re.S)]
        rows.append(" | ".join(cells))
    return rows


for path in sys.argv[1:]:
    nb = json.load(open(path))
    print(f"===== {path}")
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        for o in c.get("outputs", []):
            t = o.get("output_type")
            if t == "stream":
                print(f"[{i}] " + "".join(o["text"]).rstrip())
            elif t == "error":
                print(f"[{i}] ERROR {o['ename']}: {o['evalue']}")
            elif "data" in o:
                d = o["data"]
                if "text/html" in d and "<table" in "".join(d["text/html"]):
                    print(f"[{i}] TABLE")
                    for r in table_rows("".join(d["text/html"])):
                        print("     " + r[:220])
                elif "image/png" in d:
                    print(f"[{i}] IMAGE")
                elif "text/plain" in d:
                    print(f"[{i}] " + "".join(d["text/plain"])[:600])
