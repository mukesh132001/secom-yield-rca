# Semiconductor Yield Root-Cause Analysis (UCI SECOM)

Finds which process signals drive failed lots on a real semiconductor production line, sets up SPC control charts for them, and predicts failures before final test.

**Dataset:** [UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom). It has 1,567 production lots × 590 sensor/process signals, each lot labeled pass or fail, and about 6.6% of lots fail.

## Pipeline

| Step | What | Why it matters in a fab |
|---|---|---|
| 1. Ingest | Load lots and measurements into SQLite (long table, indexed) | Repeatable data mining on a relational database |
| 2. SQL mining | Yield by month; per-signal missing rate, distinct count, pass/fail means | Line yield reporting and data-quality screening |
| 3. Clean | Drop signals that are >50% missing or constant; fill gaps with the median | Removes dead or broken sensors |
| 4. Root cause | Mann-Whitney U test on each signal (pass vs fail), with Benjamini-Hochberg FDR correction | Systematic root-cause analysis without false positives from testing 590 signals |
| 5. SPC | X̄ ± 3σ control charts, with limits from pass lots, for the top 3 signals | Flags excursions and shows the fail rate when a signal goes out of control |
| 6. Predict | Random Forest with balanced class weights and stratified 5-fold CV; signals re-selected inside each fold (no leakage) | Catches failing lots early and quantifies the improvement |

## Run

```bash
pip install -r requirements.txt
python secom_yield_rca.py      # downloads data automatically, ~30 s
```

## Outputs (`results/`)

- `metrics.json`: all headline numbers
- `signal_ranking.csv`: every signal with its p-value, FDR q-value and effect size (σ shift)
- `top_signals.png`, `control_charts.png`
- `yield_by_month.csv`, `rf_feature_importance.csv`

## Results

_Fill in from `metrics.json` after running._

- Signals: 590 raw → **__** after cleaning
- Significant signals (q < 0.01): **__**. Top 3: **__, __, __**
- Fail prediction (top 15 signals): ROC-AUC **__**, catching **__%** of failed lots while flagging **__%** of all lots
