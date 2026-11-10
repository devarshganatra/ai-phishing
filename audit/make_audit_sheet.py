"""Build the blinded human label audit (Review 2, Critical item 1).

Draws a stratified sample of matched-pair emails, shuffles them, strips every hint of which
condition they came from, and writes:
  audit/audit_sheet_BLANK.xlsx   give one copy to each annotator (they never see the key)
  audit/KEY_do_not_share.csv     item -> condition, generator, scenario, sample_id

Sample (90 emails): 60 B_hard (2 per generator x scenario cell, all 30 cells),
20 P and 10 B_easy as reference items, so annotators are not shown only one kind of email.

  python audit/make_audit_sheet.py
"""
import os

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SEED = 20261110

INSTRUCTIONS = [
    'Email label audit: instructions',
    '',
    'You will read 90 workplace emails. Some are phishing (a scam trying to steal credentials, money or data);',
    'some are legitimate (a genuine message from the organisation). Many legitimate ones deliberately look urgent',
    'or official, so urgency alone does not make an email phishing.',
    '',
    'For each email, imagine it arrived in your work inbox from the sender it claims to be from.',
    'Decide whether the REQUEST in the email is malicious or genuine. Judge only what is written.',
    'Links and names are invented, so do not try to open them; but you may judge whether a link',
    'or contact looks like it belongs to the organisation that supposedly sent the email.',
    '',
    'Fill in, for every row of the "Emails" sheet:',
    '  Q1 Verdict      Phishing / Legitimate / Cannot tell   (use the drop-down)',
    '  Q2 Suspicion    1 = clearly safe ... 5 = clearly a scam (drop-down)',
    '  Q3 Note         optional; please add one short reason when you choose "Cannot tell"',
    '',
    'Rules:',
    '  - Work alone. Do not discuss the emails with anyone until you have finished.',
    '  - Do not use AI tools, search engines or link checkers.',
    '  - Go in order; do not go back and change earlier answers.',
    '  - There is no time limit. It takes about 60-90 minutes; a break half-way is fine.',
    '  - Fill your name or initials in cell B1 of the "Emails" sheet.',
    '',
    'Save the file as audit_<your initials>.xlsx and send it back. Thank you!',
]


def sample(mpg):
    rng = np.random.default_rng(SEED)
    parts = []
    bh = mpg[mpg['condition'] == 'B_hard']
    for _, cell in bh.groupby(['generator_model', 'scenario_id']):
        parts.append(cell.sample(2, random_state=int(rng.integers(1e9))))
    for cond, n in [('P', 20), ('B_easy', 10)]:
        sub = mpg[mpg['condition'] == cond]
        # spread over generators: equal share per generator, rest random
        per = sub.groupby('generator_model', group_keys=False).apply(
            lambda g: g.sample(n // 3, random_state=int(rng.integers(1e9))))
        rest = sub.drop(per.index).sample(n - len(per), random_state=int(rng.integers(1e9)))
        parts += [per, rest]
    s = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    s.insert(0, 'item', [f'E{i + 1:03d}' for i in range(len(s))])
    return s


def write_sheet(s, path):
    wb = Workbook()
    ins = wb.active
    ins.title = 'Instructions'
    for i, line in enumerate(INSTRUCTIONS, 1):
        c = ins.cell(row=i, column=1, value=line)
        c.font = Font(name='Calibri', size=12, bold=(i == 1))
    ins.column_dimensions['A'].width = 115

    ws = wb.create_sheet('Emails')
    ws['A1'] = 'Annotator:'
    ws['A1'].font = Font(bold=True)
    ws['B1'].fill = PatternFill('solid', fgColor='FFF2CC')
    head = ['Item', 'Email', 'Q1 Verdict', 'Q2 Suspicion (1-5)', 'Q3 Note (optional)']
    for j, h in enumerate(head, 1):
        c = ws.cell(row=3, column=j, value=h)
        c.font = Font(bold=True, color='FFFFFF')
        c.fill = PatternFill('solid', fgColor='2C5F8D')
        c.alignment = Alignment(vertical='center', wrap_text=True)
    widths = {'A': 7, 'B': 95, 'C': 15, 'D': 12, 'E': 30}
    for k, w in widths.items():
        ws.column_dimensions[k].width = w
    dv1 = DataValidation(type='list', formula1='"Phishing,Legitimate,Cannot tell"', allow_blank=True)
    dv2 = DataValidation(type='list', formula1='"1,2,3,4,5"', allow_blank=True)
    ws.add_data_validation(dv1)
    ws.add_data_validation(dv2)
    for i, r in s.iterrows():
        row = 4 + i
        ws.cell(row=row, column=1, value=r['item']).alignment = Alignment(vertical='top')
        c = ws.cell(row=row, column=2, value=str(r['text']).strip())
        c.alignment = Alignment(wrap_text=True, vertical='top')
        c.font = Font(name='Calibri', size=11)
        for col in (3, 4, 5):
            ws.cell(row=row, column=col).fill = PatternFill('solid', fgColor='FFF2CC')
            ws.cell(row=row, column=col).alignment = Alignment(vertical='top', wrap_text=True)
        dv1.add(f'C{row}')
        dv2.add(f'D{row}')
        ws.row_dimensions[row].height = min(409, 15 * (str(r['text']).count('\n') + len(str(r['text'])) // 95 + 2))
    ws.freeze_panes = 'C4'
    wb.active = 1
    wb.save(path)


def main():
    mpg = pd.read_csv(os.path.join(ROOT, 'mpg_emails.csv'))
    s = sample(mpg)
    write_sheet(s, os.path.join(HERE, 'audit_sheet_BLANK.xlsx'))
    s[['item', 'sample_id', 'condition', 'generator_model', 'scenario_id', 'label']].to_csv(
        os.path.join(HERE, 'KEY_do_not_share.csv'), index=False)
    print(s['condition'].value_counts().to_dict(), '| B_hard cells:',
          s[s.condition == 'B_hard'].groupby(['generator_model', 'scenario_id']).ngroups)
    print('wrote audit/audit_sheet_BLANK.xlsx and audit/KEY_do_not_share.csv')


if __name__ == '__main__':
    main()
