# Book And Chunk Count Statistics

This directory contains aggregate word-count, token-count, and (for English HT
and MT) sentence-count summaries. It does not contain book text.

| Path | Contents |
| --- | --- |
| `human_translation_counts/` | Aggregate word, token, and sentence counts computed from controlled-access human-translation chunks. |
| `machine_translation_counts/` | Aggregate word, token, and sentence counts computed from public MT chunks. |
| `source_text_counts/` | Aggregate word and token counts computed from controlled-access source chunks. |
| `pipelines/` | Word and token counts for the retained P1/P2/P3 public MT outputs. |

Scripts in this directory can refresh these summaries when the controlled-access
inputs are available locally.

