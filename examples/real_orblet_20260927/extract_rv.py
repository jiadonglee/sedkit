"""Download the published BEBOP VI table D5 and retain all RV columns."""
from pathlib import Path
import re,csv
import requests
from pypdf import PdfReader

HERE=Path(__file__).parent
PAPER_URL='https://pure-oai.bham.ac.uk/ws/files/252320727/SairamL2024BEBOP-VI.pdf'

def main():
    path=HERE/'paper.pdf'
    if not path.exists():
        response=requests.get(PAPER_URL,timeout=60)
        response.raise_for_status();path.write_bytes(response.content)
    text='\n'.join(page.extract_text(extraction_mode='layout') for page in PdfReader(path).pages)
    section=text.split('Ta    b    l    e  D5.')[1].split('Ta    b    l    e  D6.')[0]
    rows=[]
    for line in section.splitlines():
        if re.match(r'\s*24[56]\d{4}\.\d+',line):
            values=[float(x) for x in re.sub(r'\s*\.\s*','.',line.replace('−','-')).split()]
            assert len(values)==7
            rows.append(values)
    assert len(rows)==52
    with (HERE/'hd195987_rv.csv').open('w') as stream:
        writer=csv.writer(stream)
        writer.writerow(['bjd','rv1_kms','err1_kms','oc1_paper_kms','rv2_kms','err2_kms','oc2_paper_kms'])
        writer.writerows(rows)

if __name__=='__main__':main()
