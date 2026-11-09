import re
keys = {k for k in open(r'C:/Users/HP/OneDrive/Desktop/Dev/harness/llm_8gb_bit_costs/paper/.cited_keys.txt', encoding='utf-8').read().split() if k}
src = open(r'C:/Users/HP/OneDrive/Desktop/Dev/harness/llm_8gb_bit_costs/paper/refs.bib', encoding='utf-8').read()
out = []
for m in re.finditer(r'@(\w+)\{([^,]+),', src):
    key = m.group(2).strip()
    if key not in keys:
        continue
    i = m.end()
    depth = 1
    j = i
    while depth > 0 and j < len(src):
        if src[j] == '{': depth += 1
        elif src[j] == '}': depth -= 1
        j += 1
    out.append(src[m.start():j])
open(r'C:/Users/HP/OneDrive/Desktop/Dev/harness/llm_8gb_bit_costs/paper/jsa/revamp/cited_raw.bib', 'w', encoding='utf-8').write('\n\n'.join(out))
print(f"wrote {len(out)} raw entries to jsa/revamp/cited_raw.bib")