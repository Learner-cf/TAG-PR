# L1 Embedding Analysis Summary

- t-SNE unavailable; used PCA 2D fallback for visualization.

## Key findings
- Text↔Aerial separability margin (cosine distance): 0.282714.
- Text↔Ground separability margin (cosine distance): 0.338742.
- Better text alignment by separability margin: ground domain.
- Retrieval text_aerial: R@1=41.09, R@5=66.86, R@10=76.79, mAP=34.31.
- Retrieval text_ground: R@1=52.23, R@5=78.44, R@10=86.83, mAP=44.70.
- Collapse diagnostics suggest possible collapse.
- Analysis is restricted to text↔aerial and text↔ground; no aerial↔ground shared-ID claim is made.
