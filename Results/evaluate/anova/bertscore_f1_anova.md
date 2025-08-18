# ANOVA for bertscore_f1

**ANOVA type:** Type-3

**Model formula**  

`bertscore_f1 ~ C(system) + C(learning_rate) + C(lora_rank) + C(year_group) + C(learning_rate):C(lora_rank) + C(year_group):C(system)`


**Active factors (varying in data):** system, learning_rate, lora_rank, year_group

**Included interactions:** C(learning_rate):C(lora_rank), C(year_group):C(system)


**ANOVA table with effect sizes**

|                               |   sum_sq |   df |            F |     PR(>F) |   mean_sq |   eta_sq |   omega_sq |
|:------------------------------|---------:|-----:|-------------:|-----------:|----------:|---------:|-----------:|
| Intercept                     | 4.79173  |    1 | 60842.2      |   0        |  4.79173  | 0.998612 |   0.998579 |
| C(system)                     | 2.8e-05  |    1 |     0.349923 |   0.55757  |  2.8e-05  | 6e-06    |  -1.1e-05  |
| C(learning_rate)              | 0.000673 |    1 |     8.54743  |   0.005735 |  0.000673 | 0.00014  |   0.000124 |
| C(lora_rank)                  | 3e-06    |    1 |     0.036408 |   0.849665 |  3e-06    | 1e-06    |  -1.6e-05  |
| C(year_group)                 | 0.00259  |    2 |    16.4413   |   7e-06    |  0.001295 | 0.00054  |   0.000507 |
| C(learning_rate):C(lora_rank) | 0.000282 |    1 |     3.58228  |   0.065838 |  0.000282 | 5.9e-05  |   4.2e-05  |
| C(year_group):C(system)       | 1.5e-05  |    2 |     0.092794 |   0.911582 |  7e-06    | 3e-06    |  -3e-05    |
| Residual                      | 0.003072 |   39 |   nan        | nan        |  7.9e-05  | 0.00064  |   0        |


**Notes**

- η² (`eta_sq`) = proportion of total variance explained (biased upward).
- ω² (`omega_sq`) = less biased effect size; prefer for reporting.


*Full CSV saved to:* `evaluate/anova/bertscore_f1_anova.csv`
