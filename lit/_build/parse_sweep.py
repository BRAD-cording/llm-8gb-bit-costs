"""Repair-and-parse the truncated literature-sweep capture into a clean JSON.

The captured job output ends mid-object (A4, paper AMQ).  We cut the payload at
the last complete paper object so that the remaining structure is valid JSON,
and record the partial object separately so it can be reported, not guessed.
"""
import json
import pathlib
import re

SRC = pathlib.Path(
    r"C:\Users\HP\AppData\Local\Temp\dsh-spill-PR4qpJ\session-e068df66cbdf"
    r"\a07dfc6b9c3f-job_output.txt"
)
OUT = pathlib.Path(__file__).with_name("sweep.json")
PARTIAL = pathlib.Path(__file__).with_name("partial_amq.json")

raw = SRC.read_text(encoding="utf-8")
start = raw.index("{\n  \"areas\"")
body = raw[start:]

# Everything from the AMQ object onward is incomplete: drop it.
cut = body.index('"title": "AMQ:')
obj_start = body.rindex("\n        {", 0, cut) + 1  # opening of the AMQ paper object
head = body[: obj_start].rstrip()
partial = body[obj_start:]

# head now ends with `...},\n      ` style; normalise the tail.
head = re.sub(r",\s*$", "", head)
repaired = head + "\n      ]\n    }\n  ]\n}"

data = json.loads(repaired)
OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
PARTIAL.write_text(partial, encoding="utf-8")

print("areas:", len(data["areas"]))
total = 0
for a in data["areas"]:
    print(f'  {a["area"][:2]} sat={a["saturation"]!r} papers={len(a["papers"])} '
          f'area_gaps={len(a.get("area_gaps", []))} oq={len(a.get("open_questions", []))}')
    total += len(a["papers"])
print("papers:", total)
print("partial tail chars:", len(partial))
