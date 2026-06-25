# Book And Chunk Count Statistics

This directory contains aggregate word-count and token-count summaries. It does
not contain book text.

| Path | Contents |
| --- | --- |
| `human_translation_counts/` | Aggregate counts computed from controlled-access human-translation chunks. |
| `machine_translation_counts/` | Aggregate counts computed from public MT chunks. |
| `source_text_counts/` | Aggregate counts computed from controlled-access source chunks. |
| `pipelines/` | Counts for the retained P1/P2/P3 public MT outputs. |

Scripts in this directory can refresh these summaries when the controlled-access
inputs are available locally.

