lines = open('bibliografia.bib', 'r', encoding='utf-8').read().splitlines()
out = []
seen = set()
skip = False

for line in lines:
    if line.strip().startswith('@'):
        try:
            key = line.split('{')[1].split(',')[0].strip()
            if key in seen:
                skip = True
            else:
                skip = False
                seen.add(key)
                out.append(line)
        except IndexError:
            if not skip: out.append(line)
    elif not skip:
        out.append(line)

with open('bibliografia.bib', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out) + '\n')
