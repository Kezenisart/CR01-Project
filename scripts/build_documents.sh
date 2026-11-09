#!/bin/sh
# Rebuild the conference report and PDF slide deck from editable LaTeX sources.
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$PROJECT_DIR"
PYTHON_BIN=${PYTHON_BIN:-../../.venv/bin/python}
"$PYTHON_BIN" scripts/build_figures.py
"$PYTHON_BIN" scripts/build_extended_figure.py --results results/extended_100ep_3seed --output report/figures/extended_validation
"$PYTHON_BIN" scripts/build_extended_comparison.py
mkdir -p tmp/build_report tmp/build_slides output/pdf
(cd report && pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../tmp/build_report report.tex > ../tmp/build_report/compile.log)
(cd report && pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../tmp/build_report report.tex > ../tmp/build_report/compile.log)
(cd presentation && pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../tmp/build_slides slides.tex > ../tmp/build_slides/compile.log)
(cd presentation && pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../tmp/build_slides slides.tex > ../tmp/build_slides/compile.log)
"$PYTHON_BIN" - <<'PY'
from pypdf import PdfReader
report = PdfReader('tmp/build_report/report.pdf')
slides = PdfReader('tmp/build_slides/slides.pdf')
assert len(report.pages) == 5, 'Review report pagination: expected 4 content pages and 1 reference page.'
last = report.pages[-1].extract_text()
assert 'REFERENCES' in last and 'CONCLUSION' not in last, 'Review references-only page.'
assert len(slides.pages) == 15, 'Expected 12 main slides and 3 backup slides.'
for path in ['tmp/build_report/report.log', 'tmp/build_slides/slides.log']:
    log = open(path).read()
    assert 'Overfull' not in log, f'Check layout in {path}'
    assert 'There were undefined references' not in log, f'Check references in {path}'
print('Build checks passed: report 5 pages; deck 15 slides.')
PY
cp tmp/build_report/report.pdf output/pdf/CR01_report.pdf
cp tmp/build_slides/slides.pdf output/pdf/CR01_slides.pdf
