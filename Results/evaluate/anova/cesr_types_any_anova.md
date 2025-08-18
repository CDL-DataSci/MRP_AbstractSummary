# ANOVA for cesr_types_any

**ANOVA type:** Type-3

**Model formula**  

`cesr_types_any ~ C(system) + C(learning_rate) + C(lora_rank) + C(year_group) + C(learning_rate):C(lora_rank) + C(year_group):C(system)`


**Active factors (varying in data):** system, learning_rate, lora_rank, year_group

**Included interactions:** C(learning_rate):C(lora_rank), C(year_group):C(system)


**ANOVA table with effect sizes**

|                               |   sum_sq |   df |       F |   PR(>F) |   mean_sq |   eta_sq |   omega_sq |
|:------------------------------|---------:|-----:|--------:|---------:|----------:|---------:|-----------:|
| Intercept                     | 0.224133 |    1 | 2428.11 |        0 |  0.224133 | 0.496017 |   0.495711 |
| C(system)                     | 0        |    1 |    0    |        1 |  0        | 0        |  -0.000204 |
| C(learning_rate)              | 0        |    1 |    0    |        1 |  0        | 0        |  -0.000204 |
| C(lora_rank)                  | 0        |    1 |    0    |        1 |  0        | 0        |  -0.000204 |
| C(year_group)                 | 0.224133 |    2 | 1214.06 |        0 |  0.112067 | 0.496017 |   0.495507 |
| C(learning_rate):C(lora_rank) | 0        |    1 |    0    |        1 |  0        | 0        |  -0.000204 |
| C(year_group):C(system)       | 0        |    2 |    0    |        1 |  0        | 0        |  -0.000408 |
| Residual                      | 0.0036   |   39 |  nan    |      nan |  9.2e-05  | 0.007967 |   0        |


**Notes**

- η² (`eta_sq`) = proportion of total variance explained (biased upward).
- ω² (`omega_sq`) = less biased effect size; prefer for reporting.


*Full CSV saved to:* `evaluate/anova/cesr_types_any_anova.csv`
