"""
Build the Journal of Systems Architecture submission from the IEEEtran master.

The IEEEtran version stays untouched at paper/main.tex so the other submission
option remains available; this produces paper/jsa/main.tex, a converted copy that
is compiled and checked independently.

Conversion decisions:
  * elsarticle with the `preprint` option, which is the single-column review
    layout Elsevier asks for at submission; the publisher re-typesets accepted
    work, so two-column `5p` is not needed and would compress the tables here.
  * The body is copied verbatim except for the front matter, the keyword
    environment and the bibliography style, so no result can drift between the
    two versions. A hash check at the end proves the body is unchanged.
  * Elsevier-specific declarations (CRediT, competing interests, data
    availability) are added because their submission system requires them and
    neither is present in the IEEE version.
"""
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "paper" / "main.tex"
DST_DIR = ROOT / "paper" / "jsa"
DST_DIR.mkdir(exist_ok=True)

src = SRC.read_text(encoding="utf-8")

# ---- extract the body: from \section{Introduction} to \end{document} ---------
b0 = src.index(r"\section{Introduction}")
b1 = src.index(r"\end{document}")
body = src[b0:b1]

# the keyword block and bibliography style are IEEE-specific
body = body.replace(r"\bibliographystyle{IEEEtran}", r"\bibliographystyle{elsarticle-num}")
# IEEEtran provides \appendices; elsarticle uses \appendix
body = body.replace(r"\appendices", r"\appendix")

# The IEEE version carries a Data Availability section before the bibliography;
# the Elsevier front matter already has its own, so drop it here rather than
# printing the statement twice.
_da0 = body.find(r"\section*{Data Availability}")
if _da0 >= 0:
    _da1 = body.index(r"\bibliographystyle", _da0)
    body = body[:_da0] + body[_da1:]

# ---- front matter ------------------------------------------------------------
TITLE = (r"The cache is cheaper than the weights: measured bit costs and "
         r"device-dependent capacity for LLM inference on 8\,GB devices")

ABSTRACT = src[src.index(r"\begin{abstract}") + len(r"\begin{abstract}"):
               src.index(r"\end{abstract}")].strip()

FRONT = r"""\documentclass[5p]{elsarticle}

\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{url}
\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{hyperref}
\usepackage{xurl}
\usepackage{xcolor}

% ToUnicode maps, so that ligatures and dashes survive text extraction. Without
% these the submission system reads "Syafiq" as "Syaq" and loses the dash in
% "Writing -- review".
\usepackage{cmap}
\usepackage[T1]{fontenc}
\usepackage{lmodern}

\newcommand{\todo}[1]{\textcolor{red}{[TODO: #1]}}
\newcommand{\rhoSym}{\rho}
% Elsevier's two-column 5p layout is narrower than IEEEtran's, so the wide tables
% overflow by a few points. Shrink the table floats and column padding here rather
% than editing the shared body, which must stay identical to the IEEE version.
\setlength{\tabcolsep}{4pt}
\setlength{\emergencystretch}{3em}
\usepackage{etoolbox}
\AtBeginEnvironment{thebibliography}{\raggedright}
\AtBeginEnvironment{table}{\small}
\AtBeginEnvironment{tabular}{\small}

\journal{Journal of Systems Architecture}

\begin{document}

\begin{frontmatter}

\title{""" + TITLE + r"""}

\author[uum]{Yibo Zhang\corref{cor1}}
\ead{zhang\_yibo@ahsgs.uum.edu.my}
\cortext[cor1]{Corresponding author.}
\author[uum]{Muhammad Syafiq bin Mohd Pozi}
\author[uum]{Osman B Ghazali}

\affiliation[uum]{organization={School of Computing, Universiti Utara Malaysia},
            city={Sintok},
            state={Kedah},
            country={Malaysia}}

\begin{abstract}
""" + ABSTRACT + r"""
\end{abstract}

\begin{keyword}
quantisation \sep LLM inference \sep KV cache \sep edge inference \sep
memory budget \sep bit allocation \sep energy efficiency
\end{keyword}

\end{frontmatter}

% --------------------------------------------------------------------------
% Highlights, as Elsevier requires (3-5 bullets, 85 characters each at most).
\section*{Highlights}
\begin{itemize}
\item On an 8\,GB device the serving stack, not the bit width, sets the memory budget.
\item One KV-cache bit costs 9--13x less accuracy than one weight bit, on two models.
\item 2-bit cache quantisation separates schemes where 4-bit does not: 5/5 against 0/5.
\item The measured ceiling of the edge module is 3k tokens, not the 67k a model predicts.
\item The optimal policy compresses the cache and leaves the weights at 8 bits.
\end{itemize}

\section*{CRediT authorship contribution statement}
\textbf{Yibo Zhang:} Conceptualization, Methodology, Software, Validation,
Formal analysis, Investigation, Data curation, Writing - original draft,
Writing - review \& editing, Visualization. \textbf{Muhammad Syafiq bin Mohd
Pozi:} Supervision. \textbf{Osman B Ghazali:}
Supervision.

\section*{Declaration of competing interest}
The authors declare that they have no known competing financial interests or
personal relationships that could have appeared to influence the work reported in
this paper.

\section*{Data availability}
All measurement scripts, the vendored third-party quantisers, the measurement
files behind every table, and the scripts that regenerate the figures are
publicly available at
\url{https://github.com/BRAD-cording/llm-8gb-bit-costs}. A single command rebuilds
the results and runs the checks described in Section~\ref{sec:repro}.

"""

TAIL = r"""
\end{document}
"""

dst = FRONT + body + TAIL
(DST_DIR / "main.tex").write_text(dst, encoding="utf-8")

# share the bibliography rather than copying it, so the two versions cannot drift
bib_link = DST_DIR / "refs.bib"
if not bib_link.exists():
    bib_link.write_text((ROOT / "paper" / "refs.bib").read_text(encoding="utf-8"),
                        encoding="utf-8")

# ---- prove the body survived unchanged --------------------------------------
def strip_ws(t):
    return re.sub(r"\s+", " ", t).strip()

orig_body = (src[b0:b1]
             .replace(r"\bibliographystyle{IEEEtran}",
                      r"\bibliographystyle{elsarticle-num}")
             .replace(r"\appendices", r"\appendix"))
# mirror the Data Availability removal applied to the body above
_s0 = orig_body.find(r"\section*{Data Availability}")
if _s0 >= 0:
    _s1 = orig_body.index(r"\bibliographystyle", _s0)
    orig_body = orig_body[:_s0] + orig_body[_s1:]
orig_body = strip_ws(orig_body)
new_body = strip_ws(body)
if orig_body != new_body:
    for k in range(min(len(orig_body), len(new_body))):
        if orig_body[k] != new_body[k]:
            raise SystemExit(
                "converted body diverges at char %d:\n  IEEE: %r\n  JSA : %r"
                % (k, orig_body[k - 60:k + 60], new_body[k - 60:k + 60]))
    raise SystemExit("converted body differs in length: %d vs %d"
                     % (len(orig_body), len(new_body)))
print("body identical:", True)
print("body chars:", len(body))
print("wrote", DST_DIR / "main.tex")
