# Paper figures

Each figure is drawn in LaTeX itself (with the TikZ and pgfplots packages), so
the plotted numbers are written directly into these files. The script or result
table each figure's numbers come from is listed in Table IV of the paper.

| File | Figure |
|---|---|
| `fig-forecast-vs-price.tex` | Figure 1: model forecasts against the market price; the forecasts flatten even though no model sees the price. |
| `fig-brier-gap.tex` | Figure 2: the market's Brier score minus each model's, per depth bucket. |
| `fig-sorting.tex` | Figure 3: calibration error against sorting ability, for each model and the market. |
| `fig-recalibration.tex` | Figure 4: how much of each model's gap rescaling recovers. |
| `fig-shrinkage-null.tex` | Figure 5: whether the models agree only because they all flatten their forecasts against the price in the same way (they do not): real models against simulated forecasters that share only that flattening. |
| `fig-correlated-errors.tex` | Figure 6 (appendix): correlation between every pair of models' departures from the market price. |
| `fig-style.tex` | Not a figure: the colours and markers used for each model in every figure, defined once. |

Figure 7 (the prompt) is text, written directly in `../sections/99_appendix.tex`.
