"""Build paper/references.bib from the numbered reference list in manuscript.md.

For every reference the DOI is taken from the text, or found on Crossref by title (accepted only
when the returned title matches), or taken as the arXiv DOI for arXiv-only papers. The BibTeX
entry is then fetched from the DOI registry itself (Crossref or DataCite), as a reference manager
does when a DOI is pasted. References without a DOI (books, web pages, reports) are written by hand.
Writes paper/references.bib, paper/bib/refkeys.json (number -> key) and paper/bib/doi_report.csv.

  python paper/make_bib.py
"""
import csv
import difflib
import json
import os
import re
import time
import unicodedata
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
UA = {'User-Agent': 'phishing-intent-paper-bib/1.0'}


def get(url, accept=None):
    h = dict(UA)
    if accept:
        h['Accept'] = accept
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=30) as r:
                return r.read().decode('utf-8')
        except Exception as e:
            err = e
            time.sleep(2)
    raise err


def norm(t):
    t = unicodedata.normalize('NFKD', t).encode('ascii', 'ignore').decode().lower()
    return re.sub(r'[^a-z0-9]+', ' ', t).strip()


def refs():
    md = open(os.path.join(HERE, 'manuscript.md'), encoding='utf-8').read().split('\n## References', 1)[1]
    out = {}
    for line in md.splitlines():
        m = re.match(r'(?:&#91;|\\\[)(\d+)\\\]\s*(.*)', line.strip())
        if m:
            out[int(m.group(1))] = m.group(2)
    return out


def title_of(text):
    m = re.search(r'"(.+?)[,.?]?"', text)
    if m:
        return m.group(1)
    m = re.search(r'\*(.+?)\*', text)
    return m.group(1) if m else text


MANUAL = {
    21: '@book{biber1988variation,\n  author = {Biber, Douglas},\n  title = {Variation across Speech and Writing},\n  publisher = {Cambridge University Press},\n  address = {Cambridge, UK},\n  year = {1988},\n  doi = {10.1017/CBO9780511621024}\n}',
    22: '@book{cialdini2006influence,\n  author = {Cialdini, Robert B.},\n  title = {Influence: The Psychology of Persuasion},\n  note = {Revised edition},\n  publisher = {Harper Business},\n  address = {New York, NY, USA},\n  year = {2006}\n}',
    44: '@techreport{radford2019language,\n  author = {Radford, Alec and Wu, Jeffrey and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},\n  title = {Language Models are Unsupervised Multitask Learners},\n  institution = {OpenAI},\n  year = {2019}\n}',
    48: '@book{biber2009register,\n  author = {Biber, Douglas and Conrad, Susan},\n  title = {Register, Genre, and Style},\n  publisher = {Cambridge University Press},\n  address = {Cambridge, UK},\n  year = {2009},\n  doi = {10.1017/CBO9780511814358}\n}',
    51: '@misc{groq2025allam,\n  author = {{Groq}},\n  title = {{ALLaM 2 7B}},\n  howpublished = {GroqDocs},\n  url = {https://console.groq.com/docs/model/allam-2-7b},\n  note = {Accessed Nov. 2026}\n}',
    53: '@misc{groq2026qwen,\n  author = {{Groq}},\n  title = {{Qwen 3.8 27B}},\n  howpublished = {GroqDocs},\n  url = {https://console.groq.com/docs/model/qwen/qwen3.8-27b},\n  note = {Accessed Nov. 2026}\n}',
    55: '@misc{languagetool,\n  author = {{LanguageTool}},\n  title = {{LanguageTool}: Open-Source Grammar, Style and Spell Checker},\n  note = {Version 6.8},\n  url = {https://languagetool.org}\n}',
    62: '@book{efron1993bootstrap,\n  author = {Efron, Bradley and Tibshirani, Robert J.},\n  title = {An Introduction to the Bootstrap},\n  publisher = {Chapman \\& Hall/CRC},\n  address = {New York, NY, USA},\n  year = {1993},\n  doi = {10.1201/9780429246593}\n}',
    54: '@misc{openai2025gptoss,\n  author = {{OpenAI}},\n  title = {gpt-oss-120b \\& gpt-oss-20b Model Card},\n  howpublished = {arXiv:2508.10925},\n  year = {2025},\n  doi = {10.48550/arXiv.2508.10925},\n  url = {https://arxiv.org/abs/2508.10925}\n}',
    56: '@misc{honnibal2020spacy,\n  author = {Honnibal, Matthew and Montani, Ines and Van Landeghem, Sofie and Boyd, Adriane},\n  title = {{spaCy}: Industrial-Strength Natural Language Processing in {Python}},\n  publisher = {Zenodo},\n  year = {2020},\n  doi = {10.5281/zenodo.1212303},\n  url = {https://doi.org/10.5281/zenodo.1212303}\n}',
    64: '@book{cohen1988statistical,\n  author = {Cohen, Jacob},\n  title = {Statistical Power Analysis for the Behavioral Sciences},\n  edition = {2nd},\n  publisher = {Lawrence Erlbaum},\n  address = {Hillsdale, NJ, USA},\n  year = {1988},\n  doi = {10.4324/9780203771587}\n}',
}


def from_text(text):
    """Entry built from the manuscript reference itself, for venues that issue no DOI."""
    t = text.replace('\\_', '_').replace('–', '--')
    auth = t.split(', "')[0]
    names = [a.strip() for a in re.split(r',\s*(?:and\s+)?|\s+and\s+', auth) if a.strip()]
    out = []
    for a in names:
        if a == 'et al.' or a.endswith(' et al.'):
            if a != 'et al.':
                out.append(a[:-7])
            out.append('others')
        else:
            out.append(a)
    bibauth = ' and '.join(x if x == 'others' else (x.split()[-1] + ', ' + ' '.join(x.split()[:-1])) for x in out)
    title = title_of(t)
    venue = re.search(r'\*(.+?)\*', t)
    year = re.findall(r'\b(19|20)(\d\d)\b', t)
    year = ''.join(year[-1]) if year else ''
    pages = re.search(r'pp\. (\d+--\d+)', t)
    vol = re.search(r'vol\. (\d+)', t)
    first = norm(out[0].split()[-1] if out[0] != 'others' else 'ref').split()[-1]
    is_journal = venue and not re.search(r'Proc\.|Advances|Conf|Findings|Workshop|Symp', venue.group(1))
    kind, vfield = ('article', 'journal') if is_journal else ('inproceedings', 'booktitle')
    f = [f'  author = {{{bibauth}}}', f'  title = {{{title}}}']
    if venue:
        f.append(f'  {vfield} = {{{venue.group(1).replace("&", "\\&")}}}')
    if vol:
        f.append(f'  volume = {{{vol.group(1)}}}')
    if pages:
        f.append(f'  pages = {{{pages.group(1)}}}')
    f.append(f'  year = {{{year}}}')
    return '@' + kind + '{' + first + year + ',\n' + ',\n'.join(f) + '\n}'


# fields missing from a registry record, added from the publication page
PATCH = {15: 'author = {Shahriar, Sadat and Mukherjee, Arjun and Gnawali, Omprakash}, '
             'title = {Improving Phishing Detection via Psychological Trait Scoring}, '}
ASCII = {'&amp;': '\\&', '’': "'", '‘': '`', '′': "'", '–': '--', '—': '---', '“': '``', '”': "''", '\u00a0': ' '}


def crossref_find(text):
    q = urllib.parse.quote(re.sub(r'[*\\]', '', text)[:300])
    data = json.loads(get(f'https://api.crossref.org/works?rows=5&query.bibliographic={q}'))
    want = norm(title_of(text))
    best = None
    for it in data['message']['items']:
        for t in it.get('title', []):
            r = difflib.SequenceMatcher(None, want, norm(t)).ratio()
            if r >= 0.9 and (best is None or r > best[0]):
                best = (r, it['DOI'], t, it.get('type'))
    return best


def fetch_bibtex(doi):
    if doi.lower().startswith('10.48550/'):
        return get('https://doi.org/' + doi, accept='application/x-bibtex')
    try:
        return get(f'https://api.crossref.org/works/{urllib.parse.quote(doi)}/transform/application/x-bibtex')
    except Exception:
        return get('https://doi.org/' + doi, accept='application/x-bibtex')


def main():
    R = refs()
    entries, keys, report, used = [], {}, [], set()
    for n, text in sorted(R.items()):
        doi_txt = re.search(r'doi: (10\.\S+?)\.?$', text.replace('\\_', '_'))
        arx = re.search(r'arXiv:(\d{4}\.\d{4,5})', text)
        source, doi, note = '', None, ''
        if n in MANUAL:
            bib = MANUAL[n]
            source = 'manual'
            m = re.search(r'doi = \{(.+?)\}', bib)
            doi = m.group(1) if m else None
        else:
            if doi_txt:
                doi, source = doi_txt.group(1), 'doi in text'
            else:
                hit = crossref_find(text)
                time.sleep(0.3)
                if hit and not hit[1].lower().startswith('10.48550') and not (hit[3] == 'posted-content' and arx):
                    doi, source, note = hit[1], 'crossref title match', f'{hit[0]:.2f} {hit[3]}'
                elif arx:
                    doi, source = f'10.48550/arXiv.{arx.group(1)}', 'arXiv DOI'
            if doi:
                bib = fetch_bibtex(doi)
            else:
                bib, source = from_text(text), 'manuscript (venue has no DOI)'

        bib = bib.strip()
        if n in PATCH:
            bib = re.sub(r'^(@\w+\{[^,]+,\s*)', lambda m: m.group(1) + PATCH[n], bib)
        for a, b in ASCII.items():
            bib = bib.replace(a, b)
        bib = re.sub(r'month\s*=\s*([A-Za-z]+)', r'month={\1}', bib)
        key = re.match(r'@\w+\{([^,]+),', bib).group(1)
        first = norm(re.search(r'author\s*=\s*\{+([^,} ]+)', bib, re.I).group(1) if re.search(r'author\s*=', bib, re.I) else 'ref').split()[-1]
        ym = re.search(r'year\s*=\s*\{?(\d{4})', bib, re.I) or re.search(r'\b((?:19|20)\d\d)\b', text)
        year = ym.group(1) if ym else ''
        new = key if source == 'manual' else re.sub(r'[^a-z0-9]', '', first) + year
        if new.startswith('ref'):
            new = norm(title_of(text)).split()[0] + year
        while new in used:
            new += 'b' if not new[-1].isalpha() else chr(ord(new[-1]) + 1)
        used.add(new)
        bib = bib.replace('{' + key + ',', '{' + new + ',', 1)
        bib = re.sub(r'^\s*@', '@', bib)
        entries.append(f'% [{n}] {source}\n{bib}')
        keys[n] = new
        report.append({'n': n, 'key': new, 'doi': doi or '', 'source': source, 'match': note,
                       'manuscript_title': title_of(text)})
        print(f'[{n:>2}] {new:<22} {source:<22} {doi or "-"} {note}')
    with open(os.path.join(HERE, 'references.bib'), 'w', encoding='utf-8') as f:
        f.write('% References for "Beyond Accuracy". Generated by paper/make_bib.py: every DOI entry was\n'
                '% fetched from the DOI registry (Crossref/DataCite); entries without a DOI were written by hand.\n\n')
        f.write('\n\n'.join(entries) + '\n')
    json.dump(keys, open(os.path.join(HERE, 'bib', 'refkeys.json'), 'w'), indent=1)
    with open(os.path.join(HERE, 'bib', 'doi_report.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(report[0]))
        w.writeheader()
        w.writerows(report)


if __name__ == '__main__':
    main()
