# Performance

Docker stack (one uvicorn process, one worker, Postgres 16) after the demo: 10,002 locations, one 50,000-event catalogue (50,000 hazard
events and loss samples), one stress test with 5,213 damage samples. Reproduce with `make demo`, then `make load-test`.

## Reads: the exceedance curve and the book, closed loop, 10 s per level

| Concurrency | Requests | Throughput | p50 | p95 | p99 | Errors |
|---|---|---|---|---|---|---|
| 1 | 106 | 10.6 req/s | 153.4 ms | 221.4 ms | 259.5 ms | 0.00% |
| 8 | 182 | 17.6 req/s | 548.6 ms | 895.9 ms | 920.3 ms | 0.00% |
| 32 | 169 | 14.4 req/s | 1407.7 ms | 2366.9 ms | 2561.6 ms | 0.00% |

Measured while four other projects were building Docker images on the same 12-core laptop, so treat these as pessimistic. Even so, the
two endpoints are very different: the exceedance curve reads stored points and answers in 6–18 ms, while the book endpoint loads all
10,002 locations with their buildings and policies and aggregates them in Python on every call (about 160 ms). The book endpoint is
the whole cost of this table. A summary row written at import time would make it as cheap as the curve; not done, because the book only
changes on import.

## Compute (in-process, one core)

| Work | Time |
|---|---|
| Generate the book (10,000 locations) | under 0.1 s |
| Fit the climate to forty years and the damage curve to fifteen years of claims (censored Beta likelihood) | about 2 s |
| 50,000-event catalogue through 10,002 locations, losses, exceedance curves, bootstrap | 32 s |
| The truth catalogue for comparison (20,000 events, true climate, curves and coordinates) | about 11 s, cached |
| Stress test: one storm, 200 damage draws, seven landfall shifts at 40 draws, the truth at 100 | under 1 s of compute (one storm at 200 draws: 0.09 s), plus writing 5,213 damage rows |
| Evaluation: three coasts, five catalogues each at 30,000 events | about 4 minutes |

The catalogue's cost is location-event pairs, not events: vectorising alone made it slower until each event was limited to the
locations inside its track window with an expected damage above 1 in 100,000, in float32.

Not measured: the blueprint's 100,000-location target. At ten times the locations the catalogue would take about ten times as long on
one core; it is embarrassingly parallel by event chunk.
