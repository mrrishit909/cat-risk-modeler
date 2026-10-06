# Evaluation

Three coasts the demo never uses (seeds 21, 31, 41), 30,000 events per catalogue. Each row changes one
ingredient from the simulation truth (true climate, true damage curves, true coordinates) towards the model the demo builds.
Reproduce with `python -m ct.evaluate`.

## Average annual loss and probable maximum loss, against the truth

| Coast | Step | AAL | vs truth | 100-year PML | vs truth | 250-year PML | vs truth |
|---|---|---|---|---|---|---|---|
| 21 | Truth | $9.4M |  | $168.5M |  | $249.7M |  |
| 21 | + geocoded locations | $8.8M | -7% | $159.2M | -6% | $240.7M | -4% |
| 21 | + damage curve fitted to claims | $12.3M | +31% | $204.9M | +22% | $301.4M | +21% |
| 21 | + climate fitted to 40 years (the model) | $7.1M | -25% | $132.0M | -22% | $210.7M | -16% |
| 21 | Model with the naive damage curve | $28.8M | +206% | $340.8M | +102% | $463.8M | +86% |
| 21 | Burn cost: the 40 observed storms on today's book | $4.6M | -51% | worst year $86.6M | | | |
| 31 | Truth | $9.2M |  | $158.9M |  | $245.8M |  |
| 31 | + geocoded locations | $8.9M | -3% | $153.4M | -3% | $236.9M | -4% |
| 31 | + damage curve fitted to claims | $8.0M | -12% | $143.5M | -10% | $226.1M | -8% |
| 31 | + climate fitted to 40 years (the model) | $5.4M | -41% | $119.6M | -25% | $208.5M | -15% |
| 31 | Model with the naive damage curve | $45.9M | +401% | $447.9M | +182% | $588.2M | +139% |
| 31 | Burn cost: the 40 observed storms on today's book | $3.8M | -59% | worst year $57.0M | | | |
| 41 | Truth | $8.8M |  | $158.4M |  | $234.8M |  |
| 41 | + geocoded locations | $8.5M | -4% | $152.4M | -4% | $228.6M | -3% |
| 41 | + damage curve fitted to claims | $10.7M | +21% | $180.7M | +14% | $262.9M | +12% |
| 41 | + climate fitted to 40 years (the model) | $7.6M | -14% | $139.2M | -12% | $214.3M | -9% |
| 41 | Model with the naive damage curve | $77.9M | +781% | $459.9M | +190% | $548.1M | +133% |
| 41 | Burn cost: the 40 observed storms on today's book | $5.0M | -43% | worst year $76.5M | | | |

## Sampling error of the catalogue

90% interval of the 250-year PML from resampling whole years of the truth catalogue:

- coast 21: $241.1M to $256.6M (3.1% either side) at 30,000 events
- coast 31: $235.6M to $257.3M (4.4% either side) at 30,000 events
- coast 41: $227.3M to $241.6M (3.0% either side) at 30,000 events

## What forty years and fifteen years of claims recover

| Coast | Storms seen | Rate (true 0.45) | P(cat 4+) per landfall (true 0.020) | Reported claims / policies in footprints | Wood frame damage at 75 m/s (true 0.48) | Naive fit at 45 m/s (true 0.018) | Cat 4 at the largest town: model / truth |
|---|---|---|---|---|---|---|---|
| 21 | 16 | 0.40 | 0.004 | 751 / 8795 | 0.49 | 0.103 | $511.2M / $432.5M (+18%) |
| 31 | 18 | 0.45 | 0.009 | 455 / 9178 | 0.47 | 0.112 | $425.3M / $454.1M (-6%) |
| 41 | 22 | 0.55 | 0.017 | 218 / 3782 | 0.47 | 0.118 | $513.1M / $477.8M (+7%) |

Run time 260 s on one core.
