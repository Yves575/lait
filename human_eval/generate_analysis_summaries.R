# Generate plain-text paper summaries from human_eval/analysis_outputs.
#
# Run from the project root:
#   Rscript human_eval/generate_analysis_summaries.R
#
# Reads single_q1–q4 clmm model output files. For q2 and q3, if the CLMM shows a
# singular fit, falls back to single_q*_clm.txt for coefficients and estimated
# probabilities. Other questions fit a CLM for probabilities.

# Writes human_eval/analysis_outputs/part_1/single_reading/summaries/q*_summary.txt.
#
# Reads from part_1/single_reading/ordinal/ (CLMM + CLM fallback for q2, q3).

suppressPackageStartupMessages({
  library(ordinal)
  library(emmeans)
  library(dplyr)
})

`%||%` <- function(lhs, rhs) {
  if (is.null(lhs) || length(lhs) == 0) rhs else lhs
}


########### PATHS AND CONFIG ####################

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_arg[1] %||% "human_eval/generate_analysis_summaries.R"))
script_dir <- dirname(script_path)
human_eval_dir <- script_dir
source(file.path(script_dir, "analysis_output_paths.R"))
ensure_analysis_output_dirs()
data_dir <- file.path(script_dir, "data")

ordinal_rating_labels <- c(
  q1 = "Acceptability (1 unacceptable -> 5 acceptable)",
  q2 = "Smoothness (1 unsmooth -> 5 smooth)",
  q3 = "Immersion (1 interfered -> 5 supported)",
  q4 = "Continue reading (1 no -> 5 yes)"
)

question_paper_labels <- list(
  q1 = list(
    construct = "acceptability",
    top_label = "acceptable",
    lower_phrase = "MT was rated lower in acceptability than HT",
    higher_phrase = "MT was rated significantly lower than HT in acceptability",
    nonsign_phrase = "MT was not rated significantly lower than HT in acceptability"
  ),
  q2 = list(
    construct = "smoothness",
    top_label = "very smooth",
    lower_phrase = "MT was rated lower in smoothness than HT",
    higher_phrase = "MT was rated significantly less smooth than HT",
    nonsign_phrase = "MT was not rated significantly less smooth than HT"
  ),
  q3 = list(
    construct = "immersion",
    top_label = "highly immersive",
    lower_phrase = "MT was rated lower in immersion than HT",
    higher_phrase = "MT was rated significantly lower in immersion than HT",
    nonsign_phrase = "MT was not rated significantly lower in immersion than HT"
  ),
  q4 = list(
    construct = "continue reading",
    top_label = "very likely to continue",
    lower_phrase = "MT was rated lower on willingness to continue reading than HT",
    higher_phrase = "MT was rated significantly lower on willingness to continue reading than HT",
    nonsign_phrase = "MT was not rated significantly lower on willingness to continue reading than HT"
  )
)

########### HELPERS ####################

format_p <- function(p) {
  if (is.na(p)) return("NA")
  if (p < 0.001) return("< .001")
  formatted <- sprintf("%.4f", p)
  sub("^0\\.", ".", formatted)
}

format_num <- function(x, digits = 2) {
  if (is.na(x)) return("NA")
  format(round(x, digits), nsmall = digits, trim = TRUE)
}

format_pct <- function(prob) {
  paste0(round(prob * 100), "%")
}

parse_model_output <- function(path) {
  if (!file.exists(path)) {
    stop("Missing model output file: ", path)
  }

  lines <- readLines(path, warn = FALSE)
  question_idx <- which(lines == "Question")
  question_label <- if (length(question_idx) >= 1) lines[question_idx[1] + 1] else NA_character_

  coef_idx <- which(lines == "Coefficients:")
  if (length(coef_idx) == 0) {
    stop("Could not find Coefficients section in ", path)
  }

  coef_lines <- lines[(coef_idx[1] + 1):length(lines)]
  dash_idx <- which(coef_lines == "---")
  coef_end <- if (length(dash_idx) >= 1) dash_idx[1] - 1 else length(coef_lines)
  coef_lines <- coef_lines[seq_len(coef_end)]

  parse_coef_row <- function(name) {
    row <- coef_lines[grep(paste0("^", name), coef_lines)]
    if (length(row) == 0) return(NULL)
    parts <- strsplit(trimws(row[1]), "\\s+")[[1]]
    if (length(parts) < 5) return(NULL)
    p_raw <- gsub("[^0-9.eE+-]", "", parts[5])
    list(
      estimate = as.numeric(parts[2]),
      std_error = as.numeric(parts[3]),
      z_value = as.numeric(parts[4]),
      p_value = as.numeric(p_raw)
    )
  }

  ci_idx <- which(lines == "Confidence intervals")
  ci_lines <- if (length(ci_idx) >= 1) lines[(ci_idx[1] + 1):length(lines)] else character()
  ci_lines <- ci_lines[grepl("^typeMT|^orderMT-first", ci_lines)]

  parse_ci_row <- function(name) {
    row <- ci_lines[grep(paste0("^", name), ci_lines)]
    if (length(row) == 0) return(c(NA_real_, NA_real_))
    parts <- strsplit(trimws(row[1]), "\\s+")[[1]]
    c(as.numeric(parts[2]), as.numeric(parts[3]))
  }

  type_ci <- parse_ci_row("typeMT")
  order_ci <- parse_ci_row("orderMT-first")

  list(
    question_label = question_label,
    type = parse_coef_row("typeMT"),
    order = parse_coef_row("orderMT-first"),
    type_ci = type_ci,
    order_ci = order_ci,
    singular_fit = any(grepl("SINGULAR FIT", lines)),
    model_fallback = NULL,
    clmm_had_singular_fit = FALSE,
    source_file = basename(path)
  )
}

parse_clm_probabilities <- function(path, response) {
  if (!file.exists(path)) {
    stop("Missing CLM output file: ", path)
  }

  lines <- readLines(path, warn = FALSE)
  section_pat <- paste0("^Estimated marginal means \\(prob by type \\| ", response, "\\)")
  section_idx <- grep(section_pat, lines)
  if (length(section_idx) == 0) {
    stop("Could not find emmeans probability section in ", path)
  }

  prob_lines <- lines[section_idx[1]:length(lines)]
  level_blocks <- grep(paste0("^", response, " = "), prob_lines, value = FALSE)
  if (length(level_blocks) == 0) {
    stop("Could not parse rating-level probabilities in ", path)
  }

  by_type <- list(HT = numeric(), MT = numeric())
  levels <- integer()

  for (block_start in level_blocks) {
    header <- prob_lines[block_start]
    level <- as.integer(sub(paste0("^", response, " = ([0-9]+):.*"), "\\1", header))
    levels <- c(levels, level)

    data_rows <- prob_lines[(block_start + 2):(block_start + 3)]
    for (row in data_rows) {
      parts <- strsplit(trimws(row), "\\s+")[[1]]
      if (length(parts) < 2) next
      type_name <- parts[1]
      if (!type_name %in% names(by_type)) next
      by_type[[type_name]][as.character(level)] <- as.numeric(parts[2])
    }
  }

  list(
    levels = sort(unique(levels)),
    by_type = by_type
  )
}

clm_fallback_on_singular <- c("q2", "q3")

# Input paths: part_1/single_reading/ordinal/
single_clmm_paths <- list(
  q1 = OUTPUT_PATHS$p1_single_q1_clmm,
  q2 = OUTPUT_PATHS$p1_single_q2_clmm,
  q3 = OUTPUT_PATHS$p1_single_q3_clmm,
  q4 = OUTPUT_PATHS$p1_single_q4_clmm
)
single_clm_paths <- list(
  q2 = OUTPUT_PATHS$p1_single_q2_clm,
  q3 = OUTPUT_PATHS$p1_single_q3_clm
)
single_summary_paths <- list(
  q1 = OUTPUT_PATHS$p1_single_q1_summary,
  q2 = OUTPUT_PATHS$p1_single_q2_summary,
  q3 = OUTPUT_PATHS$p1_single_q3_summary,
  q4 = OUTPUT_PATHS$p1_single_q4_summary
)

resolve_question_analysis <- function(question_id, ordinal_data) {
  clmm_path <- single_clmm_paths[[question_id]]
  clmm_parsed <- parse_model_output(clmm_path)

  if (question_id %in% clm_fallback_on_singular && clmm_parsed$singular_fit) {
    clm_path <- single_clm_paths[[question_id]]
    if (!file.exists(clm_path)) {
      stop(
        "CLMM singular fit for ", question_id,
        " but missing CLM fallback file: ", clm_path,
        ". Run analyze_study_data.R first."
      )
    }
    cat(
      "  CLMM singular fit for ", question_id, "; using CLM fallback: ",
      basename(clm_path),
      "\n",
      sep = ""
    )
    clm_parsed <- parse_model_output(clm_path)
    clm_parsed$singular_fit <- TRUE
    clm_parsed$clmm_had_singular_fit <- TRUE
    clm_parsed$model_fallback <- "CLM"
    list(
      parsed = clm_parsed,
      probabilities = parse_clm_probabilities(clm_path, question_id),
      prob_source = basename(clm_path)
    )
  } else {
    list(
      parsed = clmm_parsed,
      probabilities = estimate_rating_probabilities(ordinal_data, question_id),
      prob_source = "CLM fit (same specification as analysis)"
    )
  }
}

load_ordinal_clmm_data <- function() {
  part1_path <- file.path(data_dir, "part1-study-data-full.csv")
  if (!file.exists(part1_path)) {
    stop("Missing data file: ", part1_path, ". Run export first or analyze_study_data.R.")
  }

  part1 <- read.csv(part1_path, na.strings = c("", "NA"), stringsAsFactors = FALSE)
  if (!"order" %in% names(part1)) {
    part1$order <- ifelse(part1$first_version == "HT", "HT-first", "MT-first")
  }

  normalize_reader_id <- function(value) {
    id <- tolower(trimws(as.character(value)))
    id <- sub("^humeval_", "", id)
    aliases <- c(
      "p0013_01" = "p013",
      "p013_01" = "p013",
      "p013_02" = "p013",
      "lauren_p1" = "p001",
      "p001_02" = "p001"
    )
    mapped <- aliases[id]
    mapped[is.na(mapped)] <- sub("_(01|02)$", "", id[is.na(mapped)])
    unname(mapped)
  }

  single_reading <- bind_rows(
    part1 %>%
      transmute(
        reader = factor(normalize_reader_id(participant_id)),
        book = factor(book_id),
        source_lang = factor(source_lang),
        order = factor(ifelse(first_version == "HT", "HT-first", "MT-first"), levels = c("HT-first", "MT-first")),
        version = factor(first_version),
        q1 = as.integer(first_q1),
        q2 = as.integer(first_q2),
        q3 = as.integer(first_q3),
        q4 = as.integer(first_q4)
      ),
    part1 %>%
      transmute(
        reader = factor(normalize_reader_id(participant_id)),
        book = factor(book_id),
        source_lang = factor(source_lang),
        order = factor(ifelse(second_version == "HT", "HT-first", "MT-first"), levels = c("HT-first", "MT-first")),
        version = factor(second_version),
        q1 = as.integer(second_q1),
        q2 = as.integer(second_q2),
        q3 = as.integer(second_q3),
        q4 = as.integer(second_q4)
      )
  )

  single_reading %>%
    transmute(
      type = factor(version, levels = c("HT", "MT")),
      order,
      q1 = ordered(q1, levels = 1:5),
      q2 = ordered(q2, levels = 1:5),
      q3 = ordered(q3, levels = 1:5),
      q4 = ordered(q4, levels = 1:5)
    ) %>%
    filter(!is.na(type))
}

estimate_rating_probabilities <- function(data, response) {
  model_data <- data %>%
    select(all_of(c(response, "type", "order"))) %>%
    filter(complete.cases(.))

  formula <- as.formula(paste(response, "~ type + order"))
  model <- clm(formula, data = model_data, Hess = TRUE)

  prob_emm <- suppressMessages(
    emmeans(model, as.formula(paste("~ type |", response)), mode = "prob")
  )
  prob_df <- as.data.frame(prob_emm)

  levels <- sort(as.integer(as.character(unique(prob_df[[response]]))))
  by_type <- split(prob_df, prob_df$type)

  list(
    levels = levels,
    by_type = lapply(by_type, function(rows) {
      stats::setNames(rows$prob, as.character(as.integer(as.character(rows[[response]]))))
    })
  )
}

describe_ci_strength <- function(beta, ci, p_value) {
  if (is.na(beta) || any(is.na(ci))) {
    return("Effect estimate unavailable.")
  }

  significant <- !is.na(p_value) && p_value < 0.05
  ci_excludes_zero <- (ci[1] > 0 && ci[2] > 0) || (ci[1] < 0 && ci[2] < 0)
  ht_odds_ratio <- exp(-beta)

  parts <- character()
  if (significant) {
    parts <- c(
      parts,
      paste0(
        "The result is statistically significant (p = ",
        format_p(p_value),
        " < .05)"
      )
    )
  } else if (!is.na(p_value) && p_value < 0.10) {
    parts <- c(parts, paste0("The result is marginal (p = ", format_p(p_value), ")"))
  } else {
    parts <- c(parts, paste0("The result is not significant (p = ", format_p(p_value), ")"))
  }

  if (ci_excludes_zero) {
    parts <- c(parts, "and the 95% CI for the type effect excludes zero")
  } else {
    parts <- c(parts, "and the 95% CI for the type effect includes zero")
  }

  if (significant && (ht_odds_ratio >= 2 || ht_odds_ratio <= 0.5)) {
    parts <- c(parts, "suggesting a large practical effect")
  }

  paste(paste(parts, collapse = "; "), ".", sep = "")
}

describe_order_effect <- function(order_coef) {
  if (is.null(order_coef) || is.na(order_coef$p_value)) {
    return("Reading order effect could not be parsed.")
  }

  if (order_coef$p_value < 0.05) {
    paste0(
      "Reading order was a significant predictor, beta = ",
      format_num(order_coef$estimate),
      ", p = ",
      format_p(order_coef$p_value),
      "."
    )
  } else {
    paste0(
      "Reading order was not a significant predictor, beta = ",
      format_num(order_coef$estimate),
      ", p = ",
      format_p(order_coef$p_value),
      "."
    )
  }
}

build_probability_lines <- function(probabilities, labels) {
  if (is.null(probabilities) || length(probabilities$levels) == 0) {
    return("Estimated probabilities: unavailable.")
  }

  top_level <- max(probabilities$levels)
  top_level_chr <- as.character(top_level)
  ht_top <- probabilities$by_type$HT[[top_level_chr]]
  mt_top <- probabilities$by_type$MT[[top_level_chr]]

  high_levels <- as.character(probabilities$levels[probabilities$levels >= (top_level - 1)])
  ht_high <- sum(probabilities$by_type$HT[high_levels], na.rm = TRUE)
  mt_high <- sum(probabilities$by_type$MT[high_levels], na.rm = TRUE)

  c(
    paste0("For rating ", top_level, " (", labels$top_label, "):"),
    paste0("  HT: ", format_pct(ht_top), " probability"),
    paste0("  MT: ", format_pct(mt_top), " probability"),
    "",
    paste0("For ratings ", min(high_levels), " or ", max(high_levels), " combined:"),
    paste0(
      "  HT: ",
      paste(vapply(high_levels, function(level) format_num(probabilities$by_type$HT[[level]], 4), character(1)), collapse = " + "),
      " = ",
      format_num(ht_high, 4),
      " (",
      format_pct(ht_high),
      ")"
    ),
    paste0(
      "  MT: ",
      paste(vapply(high_levels, function(level) format_num(probabilities$by_type$MT[[level]], 4), character(1)), collapse = " + "),
      " = ",
      format_num(mt_high, 4),
      " (",
      format_pct(mt_high),
      ")"
    ),
    paste0(
      "So HT has about ",
      format_pct(ht_high),
      " probability of receiving ",
      min(high_levels),
      " or ",
      max(high_levels),
      ", while MT has about ",
      format_pct(mt_high),
      "."
    )
  )
}

build_question_summary <- function(question_id, parsed, probabilities, prob_source) {
  labels <- question_paper_labels[[question_id]]
  type <- parsed$type
  order <- parsed$order

  if (is.null(type)) {
    stop("Missing typeMT coefficient for ", question_id)
  }

  ht_odds_ratio <- exp(-type$estimate)
  mt_odds_ratio <- exp(type$estimate)
  or_ci <- exp(parsed$type_ci)

  significant <- !is.na(type$p_value) && type$p_value < 0.05
  main_direction <- if (type$estimate < 0) {
    if (significant) labels$higher_phrase else labels$nonsign_phrase
  } else {
    if (significant) {
      paste0("MT was rated significantly higher than HT on ", labels$construct, ".")
    } else {
      paste0("MT was not rated significantly higher than HT on ", labels$construct, ".")
    }
  }

  order_note <- if (!is.null(order) && !is.na(order$p_value) && order$p_value >= 0.05) {
    "Reading order has NO effect."
  } else if (!is.null(order) && !is.na(order$p_value) && order$p_value < 0.05) {
    "Reading order has a significant effect."
  } else {
    "Reading order effect unavailable."
  }

  c(
    paste0(toupper(question_id), " — ", parsed$question_label %||% ordinal_rating_labels[[question_id]]),
    "",
    "Main result to communicate in paper:",
    if (type$estimate < 0) {
      labels$lower_phrase
    } else {
      paste0("MT was rated higher than HT on ", labels$construct, ".")
    },
    paste0(
      "HT has about ",
      format_num(ht_odds_ratio, 1),
      " times the odds of a higher ",
      labels$construct,
      " rating than MT (exp(",
      format_num(-type$estimate, 4),
      ") = ",
      format_num(ht_odds_ratio, 2),
      ")."
    ),
    describe_ci_strength(type$estimate, parsed$type_ci, type$p_value),
    order_note,
    "",
    "Model statistics:",
    paste0(
      main_direction,
      ", beta = ",
      format_num(type$estimate),
      ", SE = ",
      format_num(type$std_error),
      ", z = ",
      format_num(type$z_value),
      ", p = ",
      format_p(type$p_value),
      ", OR (MT vs HT) = ",
      format_num(mt_odds_ratio, 2),
      ", 95% CI [",
      format_num(or_ci[1], 2),
      ", ",
      format_num(or_ci[2], 2),
      "]."
    ),
    describe_order_effect(order),
    if (!is.null(parsed$model_fallback) && parsed$model_fallback == "CLM") {
      c(
        "",
        paste0(
          "Note: CLMM showed a singular fit; model statistics and estimated probabilities ",
          "are taken from ",
          parsed$source_file,
          "."
        )
      )
    } else if (parsed$singular_fit && !(question_id %in% clm_fallback_on_singular)) {
      c("", "Note: CLMM showed a singular fit; fixed effects are reported from the saved model output.")
    } else {
      character()
    },
    "",
    paste0("Estimated probabilities (", prob_source, ", averaged over order):"),
    build_probability_lines(probabilities, labels)
  )
}

ordinal_data <- load_ordinal_clmm_data()
all_summaries <- character()


########### part_1/single_reading/summaries/ ####################

for (question_id in names(ordinal_rating_labels)) {
  input_path <- single_clmm_paths[[question_id]]
  output_path <- single_summary_paths[[question_id]]

  cat("Summarizing", question_id, "from", input_path, "\n")
  analysis <- resolve_question_analysis(question_id, ordinal_data)
  summary_lines <- c(
    TEST_LABELS$paper_summary,
    "",
    build_question_summary(
      question_id,
      analysis$parsed,
      analysis$probabilities,
      analysis$prob_source
    )
  )

  writeLines(summary_lines, output_path)
  all_summaries <- c(all_summaries, summary_lines, "")
  cat(paste(summary_lines, collapse = "\n"), "\n\n")
}

combined_path <- OUTPUT_PATHS$p1_single_all_summaries
writeLines(all_summaries, combined_path)

cat("Summary generation complete. Outputs written to:", analysis_output_root, "\n")
