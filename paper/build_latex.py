"""Convert paper/manuscript.md into a self-contained LaTeX project and zip it.

The output mirrors the PDF: article class at 11 pt in Times (newtx), single column on US
Letter, booktabs tables with captions above, vector figures with captions below, numeric
citations through the `cite` package, and a BibTeX bibliography (references.bib, IEEEtran style)
whose keys come from paper/make_bib.py.

  python paper/make_figures.py
  python paper/make_bib.py
  python paper/build_latex.py      -> paper/latex/ and paper/Beyond_Accuracy_Intent_Recognition_latex.zip
"""
import json
import os
import re
import shutil
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'manuscript.md')
OUT_DIR = os.path.join(HERE, 'latex')
ZIP = os.path.join(HERE, 'Beyond_Accuracy_Intent_Recognition_latex.zip')
KEYS = {int(k): v for k, v in json.load(open(os.path.join(HERE, 'bib', 'refkeys.json'))).items()}

FIGURES = {1: 'fig1_workflow', 2: 'fig2_design', 3: 'fig3_architecture', 4: 'fig4_collapse',
           5: 'fig5_intent_auc', 6: 'fig6_features', 7: 'fig7_forest', 8: 'fig8_tfidf_terms',
           9: 'fig9_scenarios', 10: 'fig10_baselines'}
FIG_WIDTH = {2: 0.92, 3: 0.92, 8: 0.96}

PREAMBLE = r"""\documentclass[11pt,letterpaper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{newtxtext,newtxmath}
\usepackage[left=1.1in,right=1.1in,top=0.95in,bottom=0.9in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs,tabularx,array}
\usepackage{enumitem}
\usepackage{microtype}
\usepackage{titlesec}
\usepackage[labelfont=bf,labelsep=period,font=small,justification=justified]{caption}
\usepackage{cite}
\usepackage{url}
\usepackage[hidelinks]{hyperref}

\captionsetup[figure]{name=Fig.,skip=5pt}
\captionsetup[table]{justification=centering,skip=4pt}
\titleformat{\section}{\fontsize{12}{14}\selectfont\bfseries}{\thesection}{0.5em}{}
\titleformat{\subsection}{\fontsize{11}{13}\selectfont\bfseries\itshape}{\thesubsection}{0.5em}{}
\titlespacing*{\section}{0pt}{13pt}{4pt}
\titlespacing*{\subsection}{0pt}{9pt}{3pt}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt plus 1pt}
\setlist{itemsep=2pt,topsep=2pt,parsep=0pt}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\renewcommand{\arraystretch}{1.12}
\urlstyle{same}
\setcounter{secnumdepth}{0}
\setlength{\emergencystretch}{2em}
"""

UNICODE = [('⁻⁵', r'$^{-5}$'), ('–', '--'), ('±', r'$\pm$'), ('−', r'$-$'), ('≈', r'$\approx$'),
           ('≥', r'$\geq$'), ('≤', r'$\leq$'), ('×', r'$\times$'), ('→', r'$\rightarrow$'),
           ('Δ', r'$\Delta$'), ('·', r'$\cdot$'), ('κ', r'$\kappa$'), ('μ', r'$\mu$'), ('σ', r'$\sigma$'),
           ('ŷ', r'$\hat{y}$'), ('₁', r'$_1$'), ('₂', r'$_2$')]
SPECIAL = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
           '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
ESC = re.compile(r'\\([\\`*_{}\[\]()#+\-.!|<>~])')


def escape_plain(s):
    s = ''.join(SPECIAL.get(ch, ch) for ch in s)
    s = re.sub(r'"([^"]*)"', r"``\1''", s)
    for a, b in UNICODE:
        s = s.replace(a, b)
    return s


def tex_inline(text):
    """Markdown inline text -> LaTeX. Citations, URLs and escapes are protected first."""
    held = []

    def hold(tex):
        held.append(tex)
        return f'\x00{len(held) - 1}\x00'

    text = text.replace('&#91;', '\\[')
    text = re.sub(r'\b([BP])\\_(hard\\_R|hard|easy|R)\b',
                  lambda m: hold(f'${m.group(1)}_{{\\mathrm{{{m.group(2).replace(chr(92) + "_", ",")}}}}}$'), text)
    # citation ranges [a]–[b] and lists [a], [b] become one \cite
    text = re.sub(r'\\\[(\d+)\\\]–\\\[(\d+)\\\]',
                  lambda m: hold(r'\cite{' + ','.join(KEYS[k] for k in range(int(m.group(1)), int(m.group(2)) + 1)) + '}'),
                  text)
    text = re.sub(r'\\\[(\d+)\\\](?:, \\\[(\d+)\\\])*',
                  lambda m: hold(r'\cite{' + ','.join(KEYS[int(n)] for n in re.findall(r'\d+', m.group(0))) + '}'),
                  text)
    text = re.sub(r'https?://[^\s)]+', lambda m: hold(r'\url{' + m.group(0) + '}'), text)
    text = ESC.sub(lambda m: hold(escape_plain(m.group(1))), text)

    out = []
    for part in re.split(r'(\*\*.+?\*\*|\*[^*\s][^*]*?\*)', text):
        if not part:
            continue
        if part.startswith('**') and part.endswith('**'):
            out.append(r'\textbf{' + escape_plain(part[2:-2]) + '}')
        elif part.startswith('*') and part.endswith('*') and len(part) > 2:
            out.append(r'\textit{' + escape_plain(part[1:-1]) + '}')
        else:
            out.append(escape_plain(part))
    s = ''.join(out)
    while '\x00' in s:
        s = re.sub(r'\x00(\d+)\x00', lambda m: held[int(m.group(1))], s)
    return s


def table_tex(caption_md, rows):
    head, body = rows[0], rows[1:]
    ncol = len(head)
    longest = [max(len(r[j].strip()) for r in rows if j < len(r)) for j in range(ncol)]
    spec = ''.join('Y' if n > 16 else 'l' for n in longest)
    if 'Y' not in spec:
        spec = spec[:-1] + 'Y'
    caption = tex_inline(re.sub(r'^\*\*Table \d+\.\*\*\s*', '', caption_md))
    lines = [r'\begin{table}[!htbp]', r'\centering', r'\footnotesize', rf'\caption{{{caption}}}',
             rf'\begin{{tabularx}}{{\linewidth}}{{{spec}}}', r'\toprule',
             ' & '.join(r'\textbf{' + tex_inline(c.strip()) + '}' for c in head) + r' \\', r'\midrule']
    for r in body:
        lines.append(' & '.join(tex_inline(c.strip()) for c in r) + r' \\')
    lines += [r'\bottomrule', r'\end{tabularx}', r'\end{table}']
    return '\n'.join(lines)


def figure_tex(n, caption_md):
    caption = tex_inline(re.sub(r'^\*\*Fig\. \d+\.\*\*\s*', '', caption_md))
    w = FIG_WIDTH.get(n, 1.0)
    return '\n'.join([r'\begin{figure}[!htbp]', r'\centering',
                      rf'\includegraphics[width={w}\linewidth]{{figures/{FIGURES[n]}.pdf}}',
                      rf'\caption{{{caption}}}', r'\end{figure}'])


def main():
    lines = open(SRC, encoding='utf-8').read().splitlines()
    title = lines[0].lstrip('# ').strip()
    body_start = next(i for i, l in enumerate(lines) if l.startswith('## 1.'))
    front = [l for l in lines[1:body_start] if l.strip()]
    authors = next(l for l in front if l.startswith('**') and 'Abstract' not in l and 'Keywords' not in l)
    abstract = next(l for l in front if l.startswith('**Abstract.**'))
    keywords = next(l for l in front if l.startswith('**Keywords:**'))

    tex = [PREAMBLE, r'\begin{document}', '',
           r'\begin{center}',
           r'{\fontsize{17}{21}\selectfont\bfseries ' + tex_inline(title) + r'\par}',
           r'\vspace{12pt}',
           r'{\fontsize{12}{14}\selectfont ' + r'\hfill '.join(
               tex_inline(a.strip()) for a in authors.strip('* ').split(',')).join([r'\makebox[0.82\linewidth]{', '}']) + r'\par}',
           r'\end{center}', r'\vspace{6pt}',
           r'\begin{list}{}{\leftmargin=0.35in\rightmargin=0.35in}\small\item[]',
           tex_inline(abstract), '', tex_inline(keywords),
           r'\end{list}', r'\vspace{4pt}', '']

    body = lines[body_start:]
    i, table_caption, in_refs, refs = 0, None, False, []
    while i < len(body):
        s = body[i].strip()
        if not s:
            i += 1
            continue
        if s.startswith('```'):
            j = i + 1
            eq = []
            while not body[j].strip().startswith('```'):
                eq.append(body[j])
                j += 1
            for line in (l.strip() for l in eq if l.strip()):
                tex += [r'\begin{equation}', line, r'\end{equation}']
            tex.append('')
            i = j + 1
            continue
        if s.startswith('#'):
            level = len(s) - len(s.lstrip('#'))
            text = s.lstrip('# ').strip()
            if text == 'References':
                in_refs = True
                i += 1
                continue
            cmd = r'\section' if level == 2 else r'\subsection'
            tex += [f'{cmd}{{{tex_inline(text)}}}', '']
            i += 1
            continue
        if in_refs:
            m = re.match(r'(?:&#91;|\\\[)(\d+)\\\]\s*(.*)', s)
            if m:
                refs.append((int(m.group(1)), tex_inline(m.group(2))))
            i += 1
            continue
        if s.startswith('|'):
            rows = []
            while i < len(body) and body[i].strip().startswith('|'):
                cells = body[i].strip().strip('|').split('|')
                if not all(re.fullmatch(r'\s*:?-{3,}:?\s*', c) for c in cells):
                    rows.append(cells)
                i += 1
            tex += [table_tex(table_caption or '', rows), '']
            table_caption = None
            continue
        if re.match(r'\*\*Table \d+\.\*\*', s):
            table_caption = s
            i += 1
            continue
        m = re.match(r'\*\*Fig\. (\d+)\.\*\*', s)
        if m:
            tex += [figure_tex(int(m.group(1)), s), '']
            i += 1
            continue
        if s.startswith('- ') or re.match(r'\d+\. ', s):
            env = 'itemize' if s.startswith('- ') else 'enumerate'
            tex.append(rf'\begin{{{env}}}')
            while i < len(body) and (body[i].strip().startswith('- ') if env == 'itemize'
                                     else re.match(r'\d+\. ', body[i].strip())):
                item = re.sub(r'^(- |\d+\. )', '', body[i].strip())
                tex.append(r'\item ' + tex_inline(item))
                i += 1
            tex += [rf'\end{{{env}}}', '']
            continue
        tex += [tex_inline(s), '']
        i += 1

    tex += [r'\begingroup\small', r'\bibliographystyle{IEEEtran}', r'\bibliography{references}', r'\endgroup',
            '', r'\end{document}', '']

    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(os.path.join(OUT_DIR, 'figures'))
    with open(os.path.join(OUT_DIR, 'main.tex'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(tex))
    shutil.copy(os.path.join(HERE, 'references.bib'), OUT_DIR)
    for name in FIGURES.values():
        shutil.copy(os.path.join(HERE, 'figures', name + '.pdf'), os.path.join(OUT_DIR, 'figures'))
    with open(os.path.join(OUT_DIR, 'README.txt'), 'w') as f:
        f.write('Compile with pdflatex and BibTeX:\n'
                '  pdflatex main && bibtex main && pdflatex main && pdflatex main\n'
                'On Overleaf: New Project > Upload Project > choose this zip; the default pdfLaTeX\n'
                'compiler runs BibTeX automatically. References are in references.bib (IEEEtran style).\n'
                'All figures are vector PDFs in figures/.\n')
    with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(OUT_DIR):
            for fn in files:
                full = os.path.join(root, fn)
                z.write(full, os.path.relpath(full, OUT_DIR))
    print('wrote', os.path.join(OUT_DIR, 'main.tex'), 'and', ZIP, f'({len(refs)} references)')


if __name__ == '__main__':
    main()
