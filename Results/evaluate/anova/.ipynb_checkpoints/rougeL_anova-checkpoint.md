# ANOVA for rougeL

**ANOVA type:** Type-3

**Model formula**  

`rougeL ~ C(system) + C(learning_rate) + C(lora_rank) + C(year_group) + C(learning_rate):C(lora_rank) + C(year_group):C(system)`


**Active factors (varying in data):** system, learning_rate, lora_rank, year_group

**Included interactions:** C(learning_rate):C(lora_rank), C(year_group):C(system)


**ANOVA table with effect sizes**

|                               |   sum_sq |   df |          F |     PR(>F) |   mean_sq |   eta_sq |   omega_sq |
|:------------------------------|---------:|-----:|-----------:|-----------:|----------:|---------:|-----------:|
| Intercept                     | 2.47039  |    1 | 784.961    |   0        |  2.47039  | 0.917022 |   0.914785 |
| C(system)                     | 0.00151  |    1 |   0.479793 |   0.49262  |  0.00151  | 0.000561 |  -0.000607 |
| C(learning_rate)              | 0.021661 |    1 |   6.88273  |   0.012357 |  0.021661 | 0.008041 |   0.006864 |
| C(lora_rank)                  | 0.000183 |    1 |   0.058068 |   0.810838 |  0.000183 | 6.8e-05  |  -0.001099 |
| C(year_group)                 | 0.066946 |    2 |  10.6359   |   0.000206 |  0.033473 | 0.024851 |   0.022488 |
| C(learning_rate):C(lora_rank) | 0.010051 |    1 |   3.19363  |   0.0817   |  0.010051 | 0.003731 |   0.00256  |
| C(year_group):C(system)       | 0.000448 |    2 |   0.071124 |   0.931467 |  0.000224 | 0.000166 |  -0.002168 |
| Residual                      | 0.122739 |   39 | nan        | nan        |  0.003147 | 0.045561 |   0        |


**Notes**

- η² (`eta_sq`) = proportion of total variance explained (biased upward).
- ω² (`omega_sq`) = less biased effect size; prefer for reporting.


*Full CSV saved to:* `evaluate/anova/rougeL_anova.csv`
