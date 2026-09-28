# Saved outputs of the exploratory scripts

Each `.txt` file here is exactly what one script in the folder above printed
when it was run on the study's data. You can read the results without building
the databases or running anything.

- `<script>.txt` is the output of `analysis/exploratory/<script>.py`.
- `<script>-test-split.txt` is the same script run with `--test-split`, so on the
  held-out test rows only.

## The header of each file

The lines starting with `#` at the top say:

- which script produced the file (and with which option);
- when it was run, and the Python and numpy versions;
- the SHA-256 checksum of the two databases it read, so you can confirm that
  your rebuilt data matches. A database rebuilt from the release has different
  bytes from the original even though its contents are the same, so compare the
  numbers, not the checksums.

## Lines marked `[withheld: Kalshi data]`

We cannot redistribute Kalshi's market data (Window C). In six files, a few
lines would have shown it: Kalshi question text or event titles, or a number
based on fewer than 10 Window C markets, which could reveal individual markets.
Those lines are replaced by `[withheld: Kalshi data]`, and a note at the top of
each such file says so. Row counts are kept. With your own Kalshi data (see
`analysis/build_db.py`), running the script prints the full lines.
