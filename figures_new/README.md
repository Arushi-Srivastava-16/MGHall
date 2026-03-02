# Publication Figures

High-resolution, publication-quality figures for the CHG framework paper, posters, and presentations.

## Color Scheme

All figures use a consistent palette:

| Color | Hex | Used For |
|---|---|---|
| Blue | `#0288d1` | Graph construction / Math domain |
| Purple | `#7b1fa2` | GNN detection / Code domain |
| Orange | `#f57c00` | Proactive prediction |
| Green | `#388e3c` | Fingerprinting / Medical domain |
| Teal | `#00897b` | Additional metrics |

## Figures

| File | Description |
|---|---|
| `fig1_domain_performance.png` | F1 scores for node classification & origin detection across Math, Code, Medical (94–100%) |
| `fig2_error_propagation.png` | How hallucinations contaminate downstream steps — Math 3.87 steps avg, Code 2.74 steps avg |
| `fig4_ablation_study.png` | Impact of graph structure: CHG 95% F1 vs GCN 86.7% vs LSTM 75.5% vs Logit 49% |
| `fig5_lead_time_dist.png` | Early warning lead time distribution — avg 5.89 steps, 82% early detection rate |
| `fig6_error_types.png` | Hallucination type breakdown per domain (factual / logical / syntactic) |
| `fig7_cross_domain.png` | Cross-domain transfer heatmap — 12–25% drop off-diagonal vs in-domain 94–100% |
| `fig8_metrics_dashboard.png` | Full 6-subplot overview: detection, proactive, fingerprinting, propagation, error types, progress |
| `chg_banner.png` | GitHub repository banner |

## Specs

- **Resolution**: 300 DPI (print-ready)
- **Format**: PNG
- **Venues**: IJCAI, NeurIPS, ACL, AAAI

## Regenerate

```bash
python scripts/generate_enhanced_figures.py
```
