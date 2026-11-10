"""Build an editable Word version of paper/manuscript.md with the same look as the PDF.

Times New Roman 11 pt, single column on US Letter, centred title and authors, indented 10 pt
abstract, three-rule tables with captions above, figures (PNG from make_figures.py) with captions
below, hanging-indent references and centred page numbers.

  python paper/make_figures.py
  python paper/build_docx.py          -> paper/Beyond_Accuracy_Intent_Recognition.docx
"""
import html
import os
import re

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'manuscript.md')
OUT = os.path.join(HERE, 'Beyond_Accuracy_Intent_Recognition.docx')
FIG = os.path.join(HERE, 'figures')
FONT = 'Times New Roman'
TEXT_W = 6.3

FIGURES = {1: 'fig1_workflow', 2: 'fig2_design', 3: 'fig3_architecture', 4: 'fig4_collapse',
           5: 'fig5_intent_auc', 6: 'fig6_features', 7: 'fig7_forest', 8: 'fig8_tfidf_terms',
           9: 'fig9_scenarios', 10: 'fig10_baselines'}
FIG_WIDTH = {2: 0.92, 3: 0.92, 8: 0.96}

ESC = re.compile(r'\\([\\`*_{}\[\]()#+\-.!|<>~])')


# ----------------------------------------------------------------------------- text helpers
COND = re.compile(r'(?<=\b[BP])_(hard_R|hard|easy|R)\b')


def add_runs(par, text, size=None, italic_all=False):
    """Write markdown inline text (**bold**, *italic*, escapes, entities) into a paragraph."""
    held = []

    def hold(m):
        held.append(m.group(1))
        return f'\x00{len(held) - 1}\x00'
    text = ESC.sub(hold, text)
    for part in re.split(r'(\*\*.+?\*\*|\*[^*\s][^*]*?\*)', text):
        if not part:
            continue
        bold = part.startswith('**') and part.endswith('**')
        ital = not bold and part.startswith('*') and part.endswith('*') and len(part) > 2
        core = part[2:-2] if bold else part[1:-1] if ital else part
        core = re.sub(r'\x00(\d+)\x00', lambda m: held[int(m.group(1))], core)
        for k, piece in enumerate(COND.split(html.unescape(core))):
            if not piece:
                continue
            run = par.add_run(piece.replace('_', ',') if k % 2 else piece)
            run.font.subscript = bool(k % 2) or None
            run.bold = bold or None
            run.italic = (ital or italic_all) or None
            if size:
                run.font.size = Pt(size)
    return par


def fmt(par, size=11, align=WD_ALIGN_PARAGRAPH.JUSTIFY, before=0, after=6, line=1.15,
        left=0, right=0, first=None, keep=False):
    pf = par.paragraph_format
    pf.alignment = align
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    pf.line_spacing = line
    pf.left_indent, pf.right_indent = Inches(left), Inches(right)
    if first is not None:
        pf.first_line_indent = Inches(first)
    pf.keep_with_next = keep
    for r in par.runs:
        if r.font.size is None:
            r.font.size = Pt(size)
    return par


def border(el, side, sz):
    b = OxmlElement(f'w:{side}')
    b.set(qn('w:val'), 'single')
    b.set(qn('w:sz'), str(sz))
    b.set(qn('w:space'), '0')
    b.set(qn('w:color'), '000000')
    el.append(b)


# ----------------------------------------------------------------------------- blocks
def add_table(doc, rows):
    head, body = rows[0], rows[1:]
    t = doc.add_table(rows=len(rows), cols=len(head))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    # three-rule look: top and bottom of the table, and under the header row
    tbl_pr = t._tbl.tblPr
    borders = OxmlElement('w:tblBorders')
    border(borders, 'top', 10)
    border(borders, 'bottom', 10)
    tbl_pr.append(borders)
    for i, row in enumerate(rows):
        for j, cell_text in enumerate(row):
            cell = t.cell(i, j)
            p = cell.paragraphs[0]
            add_runs(p, cell_text.strip(), size=9)
            fmt(p, size=9, align=WD_ALIGN_PARAGRAPH.LEFT, after=0, line=1.05)
            if i == 0:
                for r in p.runs:
                    r.bold = True
                tc_pr = cell._tc.get_or_add_tcPr()
                cb = OxmlElement('w:tcBorders')
                border(cb, 'bottom', 6)
                tc_pr.append(cb)
    spacer = doc.add_paragraph()
    fmt(spacer, size=4, after=4, line=1.0)
    return t


def add_figure(doc, n, caption_md):
    p = doc.add_paragraph()
    width = TEXT_W * FIG_WIDTH.get(n, 1.0)
    p.add_run().add_picture(os.path.join(FIG, FIGURES[n] + '.png'), width=Inches(width))
    fmt(p, align=WD_ALIGN_PARAGRAPH.CENTER, before=6, after=3, line=1.0, keep=True)
    cap = doc.add_paragraph()
    add_runs(cap, caption_md, size=9.5)
    fmt(cap, size=9.5, after=10, line=1.1)


def add_equation(doc, k):
    p = doc.add_paragraph()
    p.paragraph_format.tab_stops.add_tab_stop(Inches(TEXT_W / 2), WD_TAB_ALIGNMENT.CENTER)
    p.paragraph_format.tab_stops.add_tab_stop(Inches(TEXT_W), WD_TAB_ALIGNMENT.RIGHT)
    p.add_run('\t')
    p.add_run().add_picture(os.path.join(FIG, f'eq{k}.png'), width=Inches(5.0))
    p.add_run(f'\t({k})')
    fmt(p, align=WD_ALIGN_PARAGRAPH.LEFT, before=3, after=8, line=1.0)


def page_number_footer(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    for tag, text in (('begin', None), (None, 'PAGE'), ('end', None)):
        if tag:
            el = OxmlElement('w:fldChar')
            el.set(qn('w:fldCharType'), tag)
        else:
            el = OxmlElement('w:instrText')
            el.set(qn('xml:space'), 'preserve')
            el.text = text
        run._r.append(el)
    run.font.size = Pt(10)


def setup(doc):
    s = doc.sections[0]
    s.page_width, s.page_height = Inches(8.5), Inches(11)
    s.left_margin = s.right_margin = Inches(1.1)
    s.top_margin, s.bottom_margin = Inches(0.95), Inches(0.9)
    normal = doc.styles['Normal']
    normal.font.name = FONT
    normal.font.size = Pt(11)
    normal.element.rPr.rFonts.set(qn('w:eastAsia'), FONT)
    for name in ('List Bullet', 'List Number'):
        st = doc.styles[name]
        st.font.name = FONT
        st.font.size = Pt(11)
    page_number_footer(s)


# ----------------------------------------------------------------------------- main
def main():
    lines = open(SRC, encoding='utf-8').read().splitlines()
    doc = Document()
    setup(doc)

    title = lines[0].lstrip('# ').strip()
    p = doc.add_paragraph()
    r = p.add_run(title)
    r.bold, r.font.size = True, Pt(17)
    fmt(p, size=17, align=WD_ALIGN_PARAGRAPH.CENTER, before=6, after=12, line=1.1, left=0.35, right=0.35)

    body_start = next(i for i, l in enumerate(lines) if l.startswith('## 1.'))
    front = [l for l in lines[1:body_start] if l.strip()]
    authors = next(l for l in front if l.startswith('**') and 'Abstract' not in l and 'Keywords' not in l)
    p = doc.add_paragraph()
    p.add_run('\t'.join(a.strip() for a in authors.strip('* ').split(',')))
    for x in (1.0, 3.15, 5.3):
        p.paragraph_format.tab_stops.add_tab_stop(Inches(x), WD_TAB_ALIGNMENT.CENTER)
    p.runs[0].text = '\t' + p.runs[0].text
    fmt(p, size=12, align=WD_ALIGN_PARAGRAPH.LEFT, after=14)
    for key in ('**Abstract.**', '**Keywords:**'):
        line = next(l for l in front if l.startswith(key))
        p = add_runs(doc.add_paragraph(), line, size=10)
        fmt(p, size=10, after=6 if key.startswith('**Abs') else 14, line=1.12, left=0.35, right=0.35)

    body = lines[body_start:]
    i, in_refs = 0, False
    pending_table_caption = None
    while i < len(body):
        line = body[i]
        s = line.strip()
        if not s:
            i += 1
            continue
        if s.startswith('```'):
            j = i + 1
            while not body[j].strip().startswith('```'):
                j += 1
            for k in (1, 2):
                add_equation(doc, k)
            i = j + 1
            continue
        if s.startswith('### ') or s.startswith('## '):
            level = 3 if s.startswith('### ') else 2
            text = s.lstrip('# ').strip()
            in_refs = text == 'References'
            p = doc.add_paragraph()
            r = p.add_run(html.unescape(ESC.sub(r'\1', text)))
            r.bold = True
            r.italic = level == 3 or None
            fmt(p, size=12 if level == 2 else 11, align=WD_ALIGN_PARAGRAPH.LEFT,
                before=14 if level == 2 else 9, after=4, keep=True)
            i += 1
            continue
        if s.startswith('|'):
            rows = []
            while i < len(body) and body[i].strip().startswith('|'):
                cells = [c for c in body[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'\s*:?-{3,}:?\s*', c) for c in cells):
                    rows.append(cells)
                i += 1
            if pending_table_caption:
                cap = doc.add_paragraph()
                add_runs(cap, pending_table_caption, size=9.5)
                fmt(cap, size=9.5, align=WD_ALIGN_PARAGRAPH.CENTER, before=6, after=3, keep=True)
                pending_table_caption = None
            add_table(doc, rows)
            continue
        m = re.match(r'\*\*Table \d+\.\*\*', s)
        if m:
            pending_table_caption = s
            i += 1
            continue
        m = re.match(r'\*\*Fig\. (\d+)\.\*\*', s)
        if m:
            add_figure(doc, int(m.group(1)), s)
            i += 1
            continue
        if re.match(r'^- ', s):
            p = add_runs(doc.add_paragraph(style='List Bullet'), s[2:])
            fmt(p, after=3)
            i += 1
            continue
        if re.match(r'^\d+\. ', s):
            p = add_runs(doc.add_paragraph(style='List Number'), re.sub(r'^\d+\. ', '', s))
            fmt(p, after=3)
            i += 1
            continue
        p = add_runs(doc.add_paragraph(), s, size=9.4 if in_refs else None)
        if in_refs:
            fmt(p, size=9.4, align=WD_ALIGN_PARAGRAPH.LEFT, after=2.5, line=1.05, left=0.33, first=-0.33)
        else:
            fmt(p)
        i += 1

    doc.core_properties.title = title
    doc.core_properties.author = 'Devarsh Ganatra, Alok Singh, Arrsh Tripathi'
    doc.save(OUT)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
