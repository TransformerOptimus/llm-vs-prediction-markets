# The paper

*Sorting, Not Calibration: A Price-Blind Comparison of Language Models and
Prediction Markets*, as a PDF and its LaTeX source.

| File or folder | What it holds |
|---|---|
| `main.pdf` | The paper: 17 pages, the main text and references first, then the appendix. |
| `main.tex` | The top-level LaTeX file: page setup, the short names used for models, exchanges and windows (for example `\winB` for "Window B"), the title and authors, and the order of the sections. |
| [`sections/`](sections/) | The text, one file per section. |
| [`figures/`](figures/) | The figures, drawn in LaTeX itself (TikZ/pgfplots), with their numbers written in. |
| [`tables/`](tables/) | The rows of each table. |
| `references.bib` | The bibliography. |
| `main.bbl` | The bibliography already formatted, as arXiv needs it. |
| `IEEEtran.cls` | The standard IEEE conference template, unchanged. |
| `Makefile` | Shortcuts for building. |

## Building the PDF

You need a TeX distribution with `latexmk` (for example TeX Live or MacTeX).

```sh
cd paper
make          # build main.pdf
make watch    # rebuild every time a file is saved
make clean    # remove the build files
```

## Where the numbers come from

Every number in the paper comes from a released file. Table IV (in the
appendix) names the script or result table behind each result; the scripts are
in [`../analysis/`](../analysis/) and the result tables in the main dataset (see
[`../data/README.md`](../data/README.md)).
