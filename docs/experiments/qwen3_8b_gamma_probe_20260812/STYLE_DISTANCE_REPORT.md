# Qwen3-8B StyleDistance drift-horizon report

## Result

![StyleDistance drift delay](figure_qwen3_8b_gamma_probe.png)

*Figure 4: StyleDistance drift delay. At $\Delta=.006$,
$K_{\mathrm{OPSD}}=26$ and $K_{\mathrm{TRSD}}=50$;
$\gamma_{\mathrm{style}}=1.92$.*

For the same StyleDistance threshold, the first crossings are
`K_OPSD=26` and
`K_TRSD=50`. Therefore:

$$
\gamma=\frac{K_{\mathrm{TRSD}}}{K_{\mathrm{baseline}}}
=\frac{50}{26}
=1.9231\approx 1.92\times.
$$

## Detailed StyleDistance trajectory

![Detailed StyleDistance trajectory](figure_qwen3_8b_style_distance_detailed.png)

| Method | Role | Δ | First crossing K | StyleDistance at K | Maximum before K |
|---|---|---:|---:|---:|---:|
| OPSD | baseline | 0.006 | 26 | 0.006996 | 0.005047 |
| TRSD | constrained | 0.006 | 50 | 0.007426 | 0.005162 |

The same `(K_OPSD, K_TRSD, γ)` result holds for every threshold in
`(0.005162, 0.006996]`.
The rounded declared threshold `Δ=0.006` lies inside this plateau.

## Metric

Each log response is embedded with the pinned
[`StyleDistance/styledistance`](https://aclanthology.org/2025.naacl-long.436/)
encoder. Long responses use 384-content-token windows with 64-token overlap;
normalized window embeddings are averaged and normalized again. At episode
`k`, drift is `1 − cosine similarity` between the centroid of responses from
episodes `k−7..k` and the same method's episode-1..8 centroid. Thus both
trajectories start from zero at the end of their shared eight-episode reference
period and are tested against the same absolute threshold.

The complete 57-point trajectory is in `style_distance_trajectory.csv`, the
crossing table is in `style_distance_crossing_table.csv`, and machine-readable
provenance is in `summary.json`. StyleDistance supplies the embedding model;
it does not prescribe a universal collapse threshold, so Δ and its stability
interval are reported explicitly.
