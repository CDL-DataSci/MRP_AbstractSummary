# Baseline vs DPDD — Comparison Report (Enhanced)

## Global Averages — Quality (selected)

| metric       |   baseline |     dpdd |   delta_dpdd_minus_baseline |   pct_change_% |
|:-------------|-----------:|---------:|----------------------------:|---------------:|
| rougeL       |   0.595686 | 0.637816 |                    0.04213  |           7.07 |
| bertscore_f1 |   0.933448 | 0.93851  |                    0.005062 |           0.54 |


## Global Averages — Privacy (selected)

| metric                  |   baseline |     dpdd |   delta_dpdd_minus_baseline |   pct_change_% |
|:------------------------|-----------:|---------:|----------------------------:|---------------:|
| precision               |   0.296296 | 0.333333 |                    0.037037 |           12.5 |
| recall                  |   0.296296 | 0.333333 |                    0.037037 |           12.5 |
| fpr                     |   0        | 0        |                    0        |          nan   |
| leak_rate               |   0        | 0        |                    0        |          nan   |
| copy_rate               |   0.060469 | 0.068027 |                    0.007559 |           12.5 |
| suppress_rate           |   0        | 0        |                    0        |          nan   |
| any_summary_canary_rate |   0.060469 | 0.068027 |                    0.007559 |           12.5 |
| cesr_types_any          |   0.060741 | 0.068333 |                    0.007593 |           12.5 |
| cesr_types_leak         |   0        | 0        |                    0        |          nan   |


## Conditional Macro Averages (avoid denom=0 bias)

| metric         |   baseline |   dpdd |
|:---------------|-----------:|-------:|
| precision_cond |          1 |      1 |
| recall_cond    |          1 |      1 |
| fpr_cond       |          0 |      0 |


## Per Hyperparameter Combo (means over runs)

|   learning_rate |   lora_rank |   retrieval_docs |   es_threshold | year_group   |   baseline_rougeL |   dpdd_rougeL |   base_prec |   dpdd_prec |   base_fpr |   dpdd_fpr |   base_cesr_leak |   dpdd_cesr_leak |
|----------------:|------------:|-----------------:|---------------:|:-------------|------------------:|--------------:|------------:|------------:|-----------:|-----------:|-----------------:|-----------------:|
|          0.0003 |           4 |                3 |            0.5 | pre-2018     |            0.5818 |        0.5987 |           0 |           0 |          0 |          0 |                0 |                0 |
|        nan      |         nan |              nan |          nan   |              |          nan      |      nan      |         nan |         nan |        nan |        nan |              nan |              nan |
|          0.0001 |           8 |                3 |            0.5 | post-2022    |            0.5767 |        0.5771 |           0 |           0 |          0 |          0 |                0 |                0 |
|          0.0001 |           8 |                3 |            0.5 | 2018-2022    |            0.6882 |        0.6628 |           1 |           1 |          0 |          0 |                0 |                0 |
|          0.0003 |           8 |                3 |            0.5 | post-2022    |            0.6038 |        0.5832 |           0 |           0 |          0 |          0 |                0 |                0 |
|          0.0001 |           4 |                3 |            0.5 | post-2022    |            0.5782 |        0.5742 |           0 |           0 |          0 |          0 |                0 |                0 |
|          0.0003 |           4 |                3 |            0.5 | 2018-2022    |            0.7908 |        0.8228 |           1 |           1 |          0 |          0 |                0 |                0 |
|          0.0003 |           4 |                3 |            0.5 | post-2022    |            0.5773 |        0.6654 |           0 |           0 |          0 |          0 |                0 |                0 |
|          0.0001 |           4 |                3 |            0.5 | 2018-2022    |            0.6619 |        0.6657 |           1 |           1 |          0 |          0 |                0 |                0 |
|          0.0001 |           8 |                3 |            0.5 | pre-2018     |            0.5973 |        0.6075 |           0 |           0 |          0 |          0 |                0 |                0 |
|          0.0003 |           8 |                3 |            0.5 | pre-2018     |            0.5806 |        0.5783 |           0 |           0 |          0 |          0 |                0 |                0 |
|          0.0001 |           4 |                3 |            0.5 | pre-2018     |            0.6004 |        0.596  |           0 |           0 |          0 |          0 |                0 |                0 |
|          0.0003 |           8 |                3 |            0.5 | 2018-2022    |            0.6548 |        0.7221 |           1 |           1 |          0 |          0 |                0 |                0 |