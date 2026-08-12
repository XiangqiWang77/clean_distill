# Qwen3-8B StyleDistance gamma

![StyleDistance drift delay](figure_qwen3_8b_gamma_probe.png)

*Figure 4: StyleDistance drift delay. At $\Delta=.006$,
$K_{\mathrm{OPSD}}=26$ and $K_{\mathrm{TRSD}}=50$;
$\gamma_{\mathrm{style}}=1.92$.*

See [STYLE_DISTANCE_REPORT.md](STYLE_DISTANCE_REPORT.md) for the detailed
StyleDistance trajectory and crossing table.

Reproduce with:

```bash
/home/da839/.conda/envs/TTT/bin/python \
  scripts/clean_self_distill/44_qwen8_gamma_probe.py
```
