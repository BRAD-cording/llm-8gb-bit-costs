"""Compatibility shim for the `pcre` package that gptqmodel imports.

gptqmodel (at least 4.2.5) does `import pcre` and uses it as a regex engine. The
PyPI name `pcre` is not on the mirrors reachable from the Jetson, and the older
`python-pcre` that is available has a different API --- it has no `pcre.Flag`,
which gptqmodel's `utils/cpp.py` needs. gptqmodel therefore fails at import.

Every use inside gptqmodel is plain: `compile`, `escape`, `split`, and the three
flag constants CASELESS / DOTALL / MULTILINE (enumerated with
`grep -rho "pcre\\.[A-Za-z_.]*" gptqmodel/`). The standard library covers that
surface exactly, so this file maps one onto the other.

Install on the target:

    SITE=$(python3 -c "import site;print(site.getusersitepackages())")
    cp artifacts/pcre_shim.py "$SITE/pcre.py"

This is a workaround, not a dependency. It shadows the real package on the user
site path; if a future gptqmodel release uses more of the PCRE API than the five
names above, this file must be extended rather than trusted.
"""
import re as _re

compile = _re.compile
escape = _re.escape
split = _re.split
match = _re.match
search = _re.search
sub = _re.sub
findall = _re.findall
finditer = _re.finditer
Pattern = _re.Pattern


class Flag:
    CASELESS = _re.IGNORECASE
    DOTALL = _re.DOTALL
    MULTILINE = _re.MULTILINE
    VERBOSE = _re.VERBOSE
    UNICODE = _re.UNICODE
    IGNORECASE = _re.IGNORECASE
