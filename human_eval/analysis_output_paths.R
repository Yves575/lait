# Shared output paths for human_eval statistical analysis scripts.
# Sourced by analyze_study_data.R and generate_analysis_summaries.R.
#
# Output layout mirrors analysis_outputs/:
#   _shared/
#   part_1/single_reading/{ordinal,word_count,origin_guess,ai_identification,summaries}/
#   part_1/comparison/{ordinal,word_count,ai_identification,excerpt_preference,preference_confidence,perceived_ht_preference}/
#   part_2/{chunk_preference,difficulty,word_count}/

# Get script path when this file is run directly; when sourced, use human_eval_dir
# set by the caller (analyze_study_data.R or generate_analysis_summaries.R).
if (!exists("human_eval_dir", inherits = TRUE)) {
  script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  paths_script <- if (length(script_arg) > 0) {
    normalizePath(sub("^--file=", "", script_arg[1]), mustWork = FALSE)
  } else if (interactive() && requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
    normalizePath(rstudioapi::getSourceEditorContext()$path, mustWork = FALSE)
  } else {
    normalizePath(file.path(getwd(), "human_eval", "analysis_output_paths.R"), mustWork = FALSE)
  }
  human_eval_dir <- if (basename(paths_script) == "analysis_output_paths.R") {
    dirname(paths_script)
  } else {
    normalizePath(file.path(getwd(), "human_eval"), mustWork = FALSE)
  }
}

analysis_output_root <- file.path(human_eval_dir, "analysis_outputs")

########### _SHARED ####################

SHARED_DIR <- file.path(analysis_output_root, "_shared")

########### PART 1 ####################

P1_SINGLE_ORDINAL <- file.path(analysis_output_root, "part_1", "single_reading", "ordinal")
P1_SINGLE_WORD_COUNT <- file.path(analysis_output_root, "part_1", "single_reading", "word_count")
P1_SINGLE_ORIGIN_GUESS <- file.path(analysis_output_root, "part_1", "single_reading", "origin_guess")
P1_SINGLE_AI <- file.path(analysis_output_root, "part_1", "single_reading", "ai_identification")
P1_SINGLE_SUMMARIES <- file.path(analysis_output_root, "part_1", "single_reading", "summaries")

P1_COMP_ORDINAL <- file.path(analysis_output_root, "part_1", "comparison", "ordinal")
P1_COMP_WORD_COUNT <- file.path(analysis_output_root, "part_1", "comparison", "word_count")
P1_COMP_AI <- file.path(analysis_output_root, "part_1", "comparison", "ai_identification")
P1_COMP_EXCERPT <- file.path(analysis_output_root, "part_1", "comparison", "excerpt_preference")
P1_COMP_PERCEIVED_HT <- file.path(analysis_output_root, "part_1", "comparison", "perceived_ht_preference")
P1_COMP_PREF_CONFIDENCE <- file.path(analysis_output_root, "part_1", "comparison", "preference_confidence")

########### PART 2 ####################

P2_CHUNK_PREF <- file.path(analysis_output_root, "part_2", "chunk_preference")
P2_DIFFICULTY <- file.path(analysis_output_root, "part_2", "difficulty")
P2_WORD_COUNT <- file.path(analysis_output_root, "part_2", "word_count")

########### TEST LABELS ####################

TEST_LABELS <- list(
  clmm = "Test: CLMM (cumulative link mixed model, ordinal logit)",
  clm = "Test: CLM (cumulative link model, ordinal logit)",
  lmer = "Test: LMER (linear mixed model, Gaussian)",
  glmer_binomial = "Test: GLMER (generalized linear mixed model, binomial logit)",
  glm_binomial = "Test: GLM (generalized linear model, binomial logit)",
  glm_quasibinomial = "Test: GLM (generalized linear model, quasibinomial logit)",
  binom = "Test: Exact binomial test (binom.test, H0: p = 0.5)",
  correlation = "Test: Spearman and Kendall rank correlation (cor.test)",
  descriptive = "Test: None (descriptive)",
  paper_summary = "Test: None (paper summary derived from CLMM/CLM outputs)"
)

########### OUTPUT PATHS ####################

OUTPUT_PATHS <- list(
  # _shared/
  shared_data_structure = file.path(SHARED_DIR, "data_structure.txt"),
  shared_response_variables = file.path(SHARED_DIR, "response_variables.txt"),
  shared_preference_descriptives = file.path(SHARED_DIR, "preference_descriptives.txt"),
  shared_annotator_id_map = file.path(SHARED_DIR, "annotator_id_map.csv"),

  # part_1/single_reading/
  p1_single_q1_clmm = file.path(P1_SINGLE_ORDINAL, "q1_acceptability_clmm.txt"),
  p1_single_q2_clmm = file.path(P1_SINGLE_ORDINAL, "q2_smoothness_clmm.txt"),
  p1_single_q2_clm = file.path(P1_SINGLE_ORDINAL, "q2_smoothness_clm.txt"),
  p1_single_q3_clmm = file.path(P1_SINGLE_ORDINAL, "q3_immersion_clmm.txt"),
  p1_single_q3_clm = file.path(P1_SINGLE_ORDINAL, "q3_immersion_clm.txt"),
  p1_single_q4_clmm = file.path(P1_SINGLE_ORDINAL, "q4_continue_reading_clmm.txt"),
  p1_single_q5_lmer = file.path(P1_SINGLE_WORD_COUNT, "q5_open_response_lmer.txt"),
  p1_single_q6_lmer = file.path(P1_SINGLE_WORD_COUNT, "q6_follow_up_lmer.txt"),
  p1_single_origin_glmer = file.path(P1_SINGLE_ORIGIN_GUESS, "origin_guess_glmer_binomial.txt"),
  p1_single_ai_confusion = file.path(P1_SINGLE_AI, "confusion_matrix.txt"),
  p1_single_ai_correct_binom = file.path(P1_SINGLE_AI, "correct_guess_binom.txt"),
  p1_single_ai_guessed_mt_glm = file.path(P1_SINGLE_AI, "guessed_MT_glm_binomial.txt"),
  p1_single_ai_guessed_mt_glmer = file.path(P1_SINGLE_AI, "guessed_MT_glmer_binomial.txt"),
  p1_single_ai_confidence_clm = file.path(P1_SINGLE_AI, "confidence_clm.txt"),
  p1_single_q1_summary = file.path(P1_SINGLE_SUMMARIES, "q1_summary.txt"),
  p1_single_q2_summary = file.path(P1_SINGLE_SUMMARIES, "q2_summary.txt"),
  p1_single_q3_summary = file.path(P1_SINGLE_SUMMARIES, "q3_summary.txt"),
  p1_single_q4_summary = file.path(P1_SINGLE_SUMMARIES, "q4_summary.txt"),
  p1_single_all_summaries = file.path(P1_SINGLE_SUMMARIES, "all_questions_summary.txt"),

  # part_1/comparison/
  p1_comp_preferred_overall_clmm = file.path(P1_COMP_ORDINAL, "preferred_overall_clmm.txt"),
  p1_comp_preferred_overall_clm = file.path(P1_COMP_ORDINAL, "preferred_overall_clm.txt"),
  p1_comp_smoother_clmm = file.path(P1_COMP_ORDINAL, "smoother_clmm.txt"),
  p1_comp_smoother_clm = file.path(P1_COMP_ORDINAL, "smoother_clm.txt"),
  p1_comp_q4_lmer = file.path(P1_COMP_WORD_COUNT, "q4_explanation_lmer.txt"),
  p1_comp_q7_lmer = file.path(P1_COMP_WORD_COUNT, "q7_second_response_lmer.txt"),
  p1_comp_ai_descriptives = file.path(P1_COMP_AI, "descriptives.txt"),
  p1_comp_ai_correct_binom = file.path(P1_COMP_AI, "correct_guess_binom.txt"),
  p1_comp_ai_guess_glm = file.path(P1_COMP_AI, "ai_guess_glm_binomial.txt"),
  p1_comp_ai_confidence_clm = file.path(P1_COMP_AI, "confidence_clm.txt"),
  p1_comp_stage_accuracy_glm = file.path(P1_COMP_AI, "stage_accuracy_glm_quasibinomial.txt"),
  p1_comp_excerpt_pref_glm = file.path(P1_COMP_EXCERPT, "prefer_HT_glm_binomial.txt"),
  p1_comp_excerpt_pref_glmer = file.path(P1_COMP_EXCERPT, "prefer_HT_glmer_binomial.txt"),
  p1_comp_excerpt_strength_glm = file.path(P1_COMP_EXCERPT, "preference_strength_glm_binomial.txt"),
  p1_comp_pref_conf_descriptives = file.path(P1_COMP_PREF_CONFIDENCE, "descriptives.txt"),
  p1_comp_pref_conf_correlation = file.path(P1_COMP_PREF_CONFIDENCE, "strength_confidence_correlation.txt"),
  p1_comp_pref_conf_clm = file.path(P1_COMP_PREF_CONFIDENCE, "confidence_by_strength_clm.txt"),
  p1_comp_pref_conf_strength_glm = file.path(P1_COMP_PREF_CONFIDENCE, "strength_by_confidence_glm_binomial.txt"),
  p1_comp_perceived_ht_descriptives = file.path(P1_COMP_PERCEIVED_HT, "descriptives.txt"),
  p1_comp_prefer_perceived_ht_binom = file.path(P1_COMP_PERCEIVED_HT, "prefer_perceived_HT_binom.txt"),
  p1_comp_prefer_perceived_ht_glm = file.path(P1_COMP_PERCEIVED_HT, "prefer_perceived_HT_glm_binomial.txt"),
  p1_comp_prefer_perceived_ht_glmer = file.path(P1_COMP_PERCEIVED_HT, "prefer_perceived_HT_glmer_binomial.txt"),

  # part_2/
  p2_preferred_translation_glmer = file.path(P2_CHUNK_PREF, "preferred_translation_glmer_binomial.txt"),
  p2_assignment_prop_ht_glm = file.path(P2_CHUNK_PREF, "assignment_prop_HT_glm_quasibinomial.txt"),
  p2_chunk_level_glmer = file.path(P2_CHUNK_PREF, "chunk_level_glmer_binomial.txt"),
  p2_reader_book_glmer = file.path(P2_CHUNK_PREF, "reader_book_glmer_binomial.txt"),
  p2_difficulty_clmm = file.path(P2_DIFFICULTY, "difficulty_clmm.txt"),
  p2_difficulty_clm = file.path(P2_DIFFICULTY, "difficulty_clm.txt"),
  p2_justification_lmer = file.path(P2_WORD_COUNT, "justification_lmer.txt")
)

ensure_analysis_output_dirs <- function() {
  dirs <- unique(c(
    SHARED_DIR,
    P1_SINGLE_ORDINAL,
    P1_SINGLE_WORD_COUNT,
    P1_SINGLE_ORIGIN_GUESS,
    P1_SINGLE_AI,
    P1_SINGLE_SUMMARIES,
    P1_COMP_ORDINAL,
    P1_COMP_WORD_COUNT,
    P1_COMP_AI,
    P1_COMP_EXCERPT,
    P1_COMP_PREF_CONFIDENCE,
    P1_COMP_PERCEIVED_HT,
    P2_CHUNK_PREF,
    P2_DIFFICULTY,
    P2_WORD_COUNT
  ))
  invisible(lapply(dirs, dir.create, showWarnings = FALSE, recursive = TRUE))
}

`%||%` <- function(lhs, rhs) {
  if (is.null(lhs) || length(lhs) == 0) rhs else lhs
}
