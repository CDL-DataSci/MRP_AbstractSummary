# ANOVA for any_summary_canary_rate

**ANOVA type:** Type-3

**Model formula**  

`any_summary_canary_rate ~ C(system) + C(learning_rate) + C(lora_rank) + C(year_group) + C(learning_rate):C(lora_rank) + C(year_group):C(system)`


**Active factors (varying in data):** system, learning_rate, lora_rank, year_group

**Included interactions:** C(learning_rate):C(lora_rank), C(year_group):C(system)


**ANOVA table with effect sizes**

|                               |   sum_sq |   df |    F |   PR(>F) |   mean_sq |   eta_sq |   omega_sq |
|:------------------------------|---------:|-----:|-----:|---------:|----------:|---------:|-----------:|
| Intercept                     | 0.22213  |    1 | 5200 |        0 |  0.22213  | 0.498132 |   0.497989 |
| C(system)                     | 0        |    1 |    0 |        1 |  0        | 0        |  -9.6e-05  |
| C(learning_rate)              | 0        |    1 |    0 |        1 |  0        | 0        |  -9.6e-05  |
| C(lora_rank)                  | 0        |    1 |    0 |        1 |  0        | 0        |  -9.6e-05  |
| C(year_group)                 | 0.22213  |    2 | 2600 |        0 |  0.111065 | 0.498132 |   0.497893 |
| C(learning_rate):C(lora_rank) | 0        |    1 |    0 |        1 |  0        | 0        |  -9.6e-05  |
| C(year_group):C(system)       | 0        |    2 |    0 |        1 |  0        | 0        |  -0.000192 |
| Residual                      | 0.001666 |   39 |  nan |      nan |  4.3e-05  | 0.003736 |  -0        |


**Notes**

- η² (`eta_sq`) = proportion of total variance explained (biased upward).
- ω² (`omega_sq`) = less biased effect size; prefer for reporting.


*Full CSV saved to:* `evaluate/anova/any_summary_canary_rate_anova.csv`
