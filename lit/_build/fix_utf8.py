"""Repair the UTF-8 double-encoding introduced by a PowerShell Set-Content round-trip."""
import pathlib
import sys

p = pathlib.Path(__file__).with_name("recovered_a5_a8.py")
raw = p.read_text(encoding="utf-8")
if "\u00e2\u20ac\u201d" not in raw and "\u00c3" not in raw:
    print("no mojibake found")
    sys.exit(0)
fixed = raw.lstrip("\ufeff").encode("cp1252", errors="strict").decode("utf-8", errors="strict")
p.write_text(fixed, encoding="utf-8")
print("em dashes:", fixed.count("\u2014"), "| residual mojibake:",
      any(ch in fixed for ch in ("\u00e2\u20ac", "\u00c3\u00a2", "\u00ef\u00bf\u00bd")))
