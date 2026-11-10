"""Typeset paper/manuscript.md as a single-column journal manuscript and print it to PDF.

The layout follows a conventional single-column article: Times New Roman 11 pt body, centred
title and authors, an indented 10 pt abstract, three-rule tables with captions above, and
figures with captions below. Figures come from paper/figures (make_figures.py).

  python paper/make_figures.py
  python paper/build_manuscript.py        -> paper/manuscript.html, paper/Beyond_Accuracy_Intent_Recognition.pdf
"""
import html
import os
import re
import subprocess

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'manuscript.md')
HTML_OUT = os.path.join(HERE, 'manuscript.html')
PDF_OUT = os.path.join(HERE, 'Beyond_Accuracy_Intent_Recognition.pdf')
CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'

FIGURES = {1: 'fig1_workflow', 2: 'fig2_design', 3: 'fig3_architecture', 4: 'fig4_collapse',
           5: 'fig5_intent_auc', 6: 'fig6_features', 7: 'fig7_forest', 8: 'fig8_tfidf_terms',
           9: 'fig9_scenarios', 10: 'fig10_baselines'}
FIG_WIDTH = {2: '92%', 3: '92%', 8: '96%'}

CSS = """
@page { size: letter; margin: 0.95in 1.1in 0.9in 1.1in;
        @bottom-center { content: counter(page); font-family: 'Times New Roman', serif; font-size: 10pt; } }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: 'Times New Roman', 'STIX Two Text', serif; font-size: 11pt; line-height: 1.3;
       color: #000; text-align: justify; hyphens: auto; margin: 0; }
h1.title { font-size: 17pt; font-weight: bold; text-align: center; line-height: 1.22;
           margin: 0.1in 0.35in 13pt; }
.authors { display: flex; justify-content: space-evenly; font-size: 12pt; margin: 0 0.6in 16pt; }
.abstract, .keywords { font-size: 10pt; line-height: 1.32; margin: 0 0.35in 6pt; }
.keywords { margin-bottom: 14pt; }
h2 { font-size: 12pt; font-weight: bold; margin: 15pt 0 5pt; break-after: avoid; text-align: left; }
h3 { font-size: 11pt; font-weight: bold; font-style: italic; margin: 11pt 0 4pt; break-after: avoid; text-align: left; }
p { margin: 0 0 7pt; orphans: 3; widows: 3; }
ul, ol { margin: 0 0 8pt; padding-left: 22pt; }
li { margin-bottom: 3pt; }
.tbl { break-inside: avoid; margin: 10pt 0 13pt; }
.tcap { font-size: 9.5pt; text-align: center; margin: 0 0 4pt; line-height: 1.3; }
table { border-collapse: collapse; width: 100%; font-size: 9.2pt; line-height: 1.24;
        border-top: 1.1pt solid #000; border-bottom: 1.1pt solid #000; text-align: left; hyphens: manual; }
th { font-weight: bold; border-bottom: 0.6pt solid #000; padding: 3pt 5pt; vertical-align: bottom; }
td { padding: 2.4pt 5pt; vertical-align: top; }
figure { margin: 9pt 0 13pt; break-inside: avoid; text-align: center; }
figure img { width: 100%; }
figcaption { font-size: 9.5pt; line-height: 1.3; text-align: justify; margin: 5pt 0 0; }
.eq { display: flex; align-items: center; margin: 4pt 0 10pt; break-inside: avoid; }
.eq img { width: 82%; margin: 0 auto; }
.eqn { width: 6%; text-align: right; }
.ref { font-size: 9.4pt; line-height: 1.25; padding-left: 24pt; text-indent: -24pt; margin: 0 0 2.6pt;
       text-align: left; overflow-wrap: anywhere; }
.decl p { font-size: 10pt; }
"""


def split_front(md):
    """Pull title, authors, abstract and keywords off the top; return them and the body."""
    lines = md.splitlines()
    title = lines[0].lstrip('# ').strip()
    body_start = next(i for i, l in enumerate(lines) if l.startswith('## 1.'))
    front = [l for l in lines[1:body_start] if l.strip()]
    authors = next(l for l in front if l.startswith('**') and 'Abstract' not in l and 'Keywords' not in l)
    abstract = next(l for l in front if l.startswith('**Abstract.**'))
    keywords = next(l for l in front if l.startswith('**Keywords:**'))
    return title, authors.strip('* '), abstract, keywords, '\n'.join(lines[body_start:])


COND = re.compile(r'\b(B|P)_(hard_R|hard|easy|R)\b')


def subscripts(h):
    return COND.sub(lambda m: f'{m.group(1)}<sub>{m.group(2).replace("_", ",")}</sub>', h)


def inline(md_text):
    return subscripts(markdown.markdown(md_text).removeprefix('<p>').removesuffix('</p>'))


def build_body(body_md):
    h = markdown.markdown(body_md, extensions=['tables', 'fenced_code', 'sane_lists'])

    # Table captions sit above their table; keep the pair together.
    h = re.sub(r'<p><strong>(Table \d+\.)</strong>(.*?)</p>\s*(<table>.*?</table>)',
               lambda m: f'<div class="tbl"><p class="tcap"><strong>{m.group(1)}</strong>{m.group(2)}</p>'
                         f'{m.group(3)}</div>', h, flags=re.S)

    # Figure captions become figures with the matching drawing above them.
    def fig(m):
        n = int(m.group(1))
        width = FIG_WIDTH.get(n, '100%')
        return (f'<figure><img src="figures/{FIGURES[n]}.svg" style="width:{width}">'
                f'<figcaption><strong>Fig. {n}.</strong>{m.group(2)}</figcaption></figure>')
    h = re.sub(r'<p><strong>Fig\. (\d+)\.</strong>(.*?)</p>', fig, h, flags=re.S)

    # The display equations are drawn by make_figures.py and numbered here.
    h = re.sub(r'<pre><code class="language-latex">.*?</code></pre>',
               ''.join(f'<div class="eq"><span class="eqn"></span><img src="figures/eq{k}.svg"><span class="eqn">({k})</span></div>'
                       for k in (1, 2)),
               h, flags=re.S)

    # References get a hanging indent; declarations a slightly smaller size.
    head, sep, refs = h.partition('<h2>References</h2>')
    refs = re.sub(r'<p>((?:\[|&#91;)\d+\].*?)</p>', r'<p class="ref">\1</p>', refs, flags=re.S)
    head = head.replace('<h2>Declarations</h2>', '<div class="decl"><h2>Declarations</h2>') + '</div>'
    return subscripts(head) + sep + refs


def main():
    md = open(SRC, encoding='utf-8').read()
    title, authors, abstract, keywords, body = split_front(md)
    names = ''.join(f'<span>{html.escape(a.strip())}</span>' for a in authors.split(','))
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{html.escape(title)}</title><style>{CSS}</style></head><body>
<h1 class="title">{html.escape(title)}</h1>
<div class="authors">{names}</div>
<p class="abstract">{inline(abstract)}</p>
<p class="keywords">{inline(keywords)}</p>
{build_body(body)}
</body></html>"""
    open(HTML_OUT, 'w', encoding='utf-8').write(page)
    subprocess.run([CHROME, '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
                    '--allow-file-access-from-files', f'--print-to-pdf={PDF_OUT}',
                    'file://' + HTML_OUT], check=True, capture_output=True)
    print('wrote', HTML_OUT, 'and', PDF_OUT)


if __name__ == '__main__':
    main()
