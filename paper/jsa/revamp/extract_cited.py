import re, sys
src = open('main.tex', encoding='utf-8').read()
# capture the argument of any \cite / \citep / \citet command
keys = set()
for m in re.finditer(r'\\cite[a-z]*\{([^}]*)\}', src, re.I):
    for k in m.group(1).split(','):
        k = k.strip()
        if k:
            keys.add(k)
sys.stdout.write('\n'.join(sorted(keys)) + '\n')
print(f'# total unique cited: {len(keys)}', file=sys.stderr)