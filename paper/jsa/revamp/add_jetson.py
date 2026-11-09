"""Insert the Jetson second-device block into the executed-policy subsection.

paper/main.tex is the master and uses CRLF, so the replacement is done here rather
than through the line-based editor.
"""

from pathlib import Path

MASTER = Path(r"C:\Users\HP\OneDrive\Desktop\Dev\harness\llm_8gb_bit_costs\paper\main.tex")
text = MASTER.read_bytes().decode("utf-8")
assert "\r\n" in text

ANCHOR = "Three boundaries. The scheme is llama.cpp's per-32-block packing"
assert text.count(ANCHOR) == 1
assert "label{sec:rho}" in text, "budget section label changed"

NEW = r"""\emph{A second device.} The Orin Nano Super is the unified-memory endpoint of the
same argument: 8\,GB nominal, no separate VRAM, and the budget that
Section~\ref{sec:rho} measured at $\rho=0.419$. We build the same runtime tag on
the device (sm\_87, CUDA 12.6) and load the same GGUF. On this platform the runtime
maps the weights instead of copying them into a device buffer, so the cache is what
competes for the budget.

\begin{table}[t]
\caption{The same policy on a unified-memory device: Jetson Orin Nano Super
(8\,GB, 25\,W, desktop running), llama.cpp \texttt{b12063}, Qwen3-1.7B with Q8\_0
weights. ``KV'' is the cache the runtime allocates, not a model buffer it does not
copy; an em dash means the allocation was refused in all three attempts. Prefill on
this device varies by up to $20\%$ between runs and decode by less than $5\%$; no
claim here rests on prefill.}
\label{tab:exec-orin}
\begin{tabular}{@{}lrrrr@{}}
\toprule
$L$ & cache & KV (MiB) & prefill & decode \\
\midrule
4096  & fp16  &   448 & 1{,}209 & 24.3 \\
      & q8\_0 &   238 &   989 & 24.6 \\
      & q4\_0 &   126 & 1{,}139 & 22.9 \\
\addlinespace
8192  & fp16  &   896 & 1{,}069 & 21.4 \\
      & q8\_0 &   476 & 1{,}040 & 22.0 \\
      & q4\_0 &   252 & 1{,}024 & 25.0 \\
\addlinespace
16384 & fp16  & 1{,}792 & \multicolumn{2}{c}{refused} \\
      & q8\_0 &   952 &   854 & 24.3 \\
      & q4\_0 &   504 &   876 & 22.8 \\
\addlinespace
32768 & fp16  & 3{,}584 & \multicolumn{2}{c}{refused} \\
      & q8\_0 & 1{,}904 & \multicolumn{2}{c}{refused} \\
      & q4\_0 & 1{,}008 &   648 & 24.2 \\
\bottomrule
\end{tabular}
\end{table}

Table~\ref{tab:exec-orin} is a ladder: 8k at 16 bits (896\,MiB), 16k at 8 bits
(952), 32k at 4 bits (1{,}008). Each halving of the per-token cache buys exactly one
doubling of attainable context, and the rung above each step is refused in all three
attempts (1{,}792 and 1{,}904\,MiB). What makes this a capacity result rather than
an out-of-memory result is the measured peak: the largest working cell peaks at
5{,}235 of 7{,}620\,MB, so the device is not full when it refuses a 1{,}792\,MiB
allocation. The runtime reports 4{,}482\,MiB free at that moment, and a plain CUDA
program can allocate 1{,}793, 238 and 952\,MiB in the same order --- so the refusal
belongs to the runtime's allocation pattern, which maps the weights first, and not
to the hardware. That is the paper's thesis in miniature: the budget a configuration
gets is neither the number on the box nor the number the runtime reports.

Accuracy reproduces across devices, and only the budget does not. The same quantiser
on the same GGUF costs $+0.10\%$ at 8 bits (15.1822 to 15.1968) and $+25.06\%$ at 4
bits (18.9865) on the Orin, against $+0.11\%$ and $+26.15\%$ on the laptop. The policy
that follows is therefore device-specific and computable: on this device q8\_0 at 16k
occupies the same budget as fp16 at 8k (952 against 896\,MiB), doubles the context,
and costs no measurable accuracy, which is a better trade than 4 bits at 32k --- a
third of the cache for a quarter more perplexity.

Four boundaries. The scheme is llama.cpp's per-32-block packing"""

NEW = NEW.replace("\n", "\r\n")
text = text.replace(ANCHOR, NEW, 1)

import re

text, n = re.subn(
    r"And this is one device, one runtime(\r?\n)and one weight format, which bounds the reading",
    r"And these are two devices, one runtime\1and one weight format, which bounds the reading",
    text,
)
assert n == 1, f"device-count sentence edited {n} times"
out = text
MASTER.write_bytes(out.encode("utf-8"))
print("inserted; table count", out.count("\\begin{table}"), "labels", out.count("\\label{tab:exec-orin}"))
