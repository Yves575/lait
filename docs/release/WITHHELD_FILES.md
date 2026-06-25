# Withheld Files

This file summarizes the files removed from the public GitHub branch. The exact machine-readable manifest is `withheld-files.tsv`.

- Total withheld files: 8,066
- Manifest: `docs/release/withheld-files.tsv`
- Pathspec list: `docs/release/all-withheld-removals.txt`

## Counts By Reason

| Reason | Files |
| --- | ---: |
| `controlled_access_litransproqa_text_dataset` | 7 |
| `controlled_access_raw_human_eval_or_internal_material` | 73 |
| `controlled_access_row_level_human_eval_material` | 75 |
| `controlled_access_row_level_or_alignment_text` | 7 |
| `controlled_access_run_or_alignment_material` | 6,732 |
| `controlled_access_source_or_human_translation_text` | 249 |
| `out_of_preprint_scope_detector_outputs_not_reported` | 8 |
| `out_of_preprint_scope_detector_workflow_not_reported` | 1 |
| `out_of_preprint_scope_intermediate_mt_draft` | 90 |
| `out_of_preprint_scope_internal_alignment_helpers` | 9 |
| `out_of_preprint_scope_internal_alignment_prompts` | 5 |
| `out_of_preprint_scope_internal_alignment_tests` | 9 |
| `out_of_preprint_scope_internal_chunk_review_helper` | 1 |
| `out_of_preprint_scope_internal_chunk_review_tool` | 41 |
| `out_of_preprint_scope_internal_ht_chunking_helper` | 3 |
| `out_of_preprint_scope_internal_ht_chunking_tests` | 1 |
| `out_of_preprint_scope_internal_planning_notes` | 13 |
| `out_of_preprint_scope_internal_utility_script` | 10 |
| `out_of_preprint_scope_legacy_analysis_notes` | 1 |
| `out_of_preprint_scope_metric_notebook` | 1 |
| `relocated_public_litransproqa_weight_config` | 1 |
| `out_of_preprint_scope_stale_agent_run_summary` | 2 |
| `out_of_preprint_scope_stale_analysis_build_index` | 1 |
| `out_of_preprint_scope_stale_automatic_metric_builder` | 1 |
| `out_of_preprint_scope_stale_generated_analysis_builder` | 5 |
| `out_of_preprint_scope_stale_generated_analysis_helper` | 1 |
| `out_of_preprint_scope_stale_generated_analysis_outputs` | 242 |
| `out_of_preprint_scope_stale_metric_plots` | 32 |
| `out_of_preprint_scope_stale_pipeline_count_builder` | 1 |
| `out_of_preprint_scope_standalone_q5_agreement_subproject` | 21 |
| `out_of_preprint_scope_superseded_vocab_subproject` | 8 |
| `out_of_preprint_scope_unreported_gemba_metric_details` | 30 |
| `out_of_preprint_scope_unreported_gemba_metric_tests` | 1 |
| `out_of_preprint_scope_unreported_generated_plots` | 47 |
| `out_of_preprint_scope_unreported_msft_baseline_outputs` | 31 |
| `out_of_preprint_scope_unreported_mt_model_metric_outputs` | 186 |
| `out_of_preprint_scope_unreported_mt_model_outputs` | 107 |
| `out_of_preprint_scope_unreported_mt_model_stats` | 8 |
| `out_of_preprint_scope_unused_gemba_submodule` | 2 |
| `withheld_binary_aligned_data_embeds_source_ht` | 2 |
| `withheld_from_public_release` | 1 |

## Counts By Top-Level Path

| Top-level path | Files |
| --- | ---: |
| `.gitmodules` | 1 |
| `GEMBA` | 1 |
| `INTERNAL_study-data-full-translated.json` | 1 |
| `INTERNAL_study-data-full.json` | 1 |
| `LiTransProQA` | 10 |
| `align_chunks` | 7 |
| `analysis` | 312 |
| `analysis.md` | 1 |
| `book_stats` | 8 |
| `books` | 477 |
| `data` | 2 |
| `docs` | 15 |
| `human_eval` | 82 |
| `internal_eval` | 4 |
| `par3` | 36 |
| `par3_dataset` | 341 |
| `prompts` | 5 |
| `q5_agreement` | 21 |
| `results_ai_detection` | 8 |
| `results_all_metrics` | 218 |
| `results_chunk_review_eval` | 33 |
| `results_mapped_metrics` | 3 |
| `runs` | 6,355 |
| `scripts` | 16 |
| `statistical` | 47 |
| `tests` | 11 |
| `tools` | 42 |
| `vocab_comparison_project` | 8 |

## Manual Inspection

Use `withheld-files.tsv` to inspect individual paths. If a file should be restored, confirm that it does not expose source text, human-translation text, row-level participant material, private data, intermediate draft outputs, or an out-of-scope workflow before adding it back.

Note: `LiTransProQA/datasets/QA_weights.csv` was removed from the old dataset directory but its non-text weight values were retained at `LiTransProQA/config/question_weights.csv`.
