import re, sys
keys = set(open(r'C:/Users/HP/OneDrive/Desktop/Dev/harness/llm_8gb_bit_costs/paper/.cited_keys.txt', encoding='utf-8').read().split())
keys.discard('')
src = open(r'C:/Users/HP/OneDrive/Desktop/Dev/harness/llm_8gb_bit_costs/paper/refs.bib', encoding='utf-8').read()
found = []
missing = sorted(keys)
for m in re.finditer(r'@(\w+)\{([^,]+),', src):
    key = m.group(2).strip()
    if key not in keys:
        continue
    if key in missing:
        missing.remove(key)
    i = m.end()  # right after 'key,'
    depth = 1
    j = i
    while depth > 0 and j < len(src):
        if src[j] == '{':
            depth += 1
        elif src[j] == '}':
            depth -= 1
        j += 1
    body = src[i:j-1]
    # pull the core fields for a readable citation record
    def field(f):
        mm = re.search(r'(?m)^\s*' + f + r'\s*=\s*\{(.+?)\}', body, re.S)
        return mm.group(1).strip() if mm else ''
    found.append({
        'key': key,
        'type': m.group(1),
        'title': field('title'),
        'author': field('author'),
        'year': field('year'),
        'journal': field('journal') or field('booktitle') or field('howpublished') or '',
        'doi': field('doi'),
        'url': field('url'),
        'volume': field('volume'),
        'pages': field('pages'),
    })
missing = [k for k in missing]
with open(r'C:/Users/HP/OneDrive/Desktop/Dev/harness/llm_8gb_bit_costs/paper/jsa/revamp/cited_entries.txt', 'w', encoding='utf-8') as o:
    for e in found:
        o.write(f"KEY: {e['key']} [{e['type']}]\n")
        o.write(f"  title:   {e['title']}\n")
        o.write(f"  author:  {e['author']}\n")
        o.write(f"  year:    {e['year']}\n")
        o.write(f"  venue:   {e['journal']}\n")
        o.write(f"  doi:     {e['doi']}\n")
        o.write(f"  vol/pg:  {e['volume']}/{e['pages']}\n")
        o.write(f"  url:     {e['url']}\n\n")
print(f"extracted {len(found)} entries to jsa/revamp/cited_entries.txt")
print("keys not found in refs.bib:", missing)