# Human evaluation statistical analysis.
#
# Run from the project root:
#   Rscript human_eval/analyze_study_data.R
#
# The script reads the exported CSVs in human_eval/data/, coerces response variables
# to appropriate types, fits mixed-effects models where the data support them, and
# writes model summaries to human_eval/analysis_outputs/:
#   _shared/, part_1/single_reading/, part_1/comparison/, part_2/
#
# Single-reading ordinal outcomes (1-5):
#   q1 — Acceptability (1 unacceptable -> 5 acceptable)
#   q2 — Smoothness (1 unsmooth -> 5 smooth)
#   q3 — Immersion (1 interfered -> 5 supported)
#   q4 — Continue reading (1 no -> 5 yes)

suppressPackageStartupMessages({
  library(ordinal)
  library(lme4)
  library(dplyr)
  library(emmeans)
})

`%||%` <- function(lhs, rhs) {
  if (is.null(lhs) || length(lhs) == 0) rhs else lhs
}


########### PATHS AND LOADING DATA ####################

# Get script path - works with both Rscript and interactive/VS Code
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- if (length(script_arg) > 0) {
  normalizePath(sub("^--file=", "", script_arg[1]), mustWork = FALSE)
} else if (interactive() && requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
  # Running in RStudio or VS Code
  normalizePath(rstudioapi::getSourceEditorContext()$path, mustWork = FALSE)
} else {
  # Fallback: assume running from project root
  normalizePath("human_eval/analyze_study_data.R", mustWork = FALSE)
}
script_dir <- dirname(script_path)
project_root <- normalizePath(file.path(script_dir, ".."))
data_dir <- file.path(script_dir, "data")
human_eval_dir <- script_dir
source(file.path(script_dir, "analysis_output_paths.R"))
ensure_analysis_output_dirs()

part1_path <- file.path(data_dir, "part1-study-data-full.csv")
part2_path <- file.path(data_dir, "part2-study-data-full.csv")
part2_span_path <- file.path(data_dir, "part2-span-study-data-full.csv")

part1 <- read.csv(part1_path, na.strings = c("", "NA"), stringsAsFactors = FALSE)
part2 <- read.csv(part2_path, na.strings = c("", "NA"), stringsAsFactors = FALSE)
part2_span <- read.csv(part2_span_path, na.strings = c("", "NA"), stringsAsFactors = FALSE)

if (!"user_id" %in% names(part1)) part1$user_id <- part1$participant_id
if (!"user_id" %in% names(part2)) part2$user_id <- part2$participant_id
if (!"user_id" %in% names(part2_span)) part2_span$user_id <- part2_span$participant_id

language_group_column <- Sys.getenv("LITMT_LANGUAGE_GROUP_COLUMN", "source_lang")
if (language_group_column != "source_lang") {
  if (!language_group_column %in% names(part1)) {
    stop(paste("Missing language grouping column in part1:", language_group_column))
  }
  if (!language_group_column %in% names(part2)) {
    stop(paste("Missing language grouping column in part2:", language_group_column))
  }
  if (!language_group_column %in% names(part2_span)) {
    stop(paste("Missing language grouping column in part2_span:", language_group_column))
  }

  part1$source_lang <- part1[[language_group_column]]
  part2$source_lang <- part2[[language_group_column]]
  part2_span$source_lang <- part2_span[[language_group_column]]
}

sink(file.path(output_dir, "data_structure.txt"))
cat("Part 1 structure\n")
str(part1)
cat("\nPart 1 summary\n")
print(summary(part1))
cat("\nPart 2 structure\n")
str(part2)
cat("\nPart 2 summary\n")
print(summary(part2))
cat("\nPart 2 span structure\n")
str(part2_span)
cat("\nPart 2 span summary\n")
print(summary(part2_span))
sink()


########### HELPERS ####################

capture_model_warnings <- function(expr) {
  warnings <- character()
  value <- withCallingHandlers(
    expr,
    warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  list(value = value, warnings = unique(warnings))
}

has_enough_levels <- function(data, columns) {
  all(vapply(columns, function(column) {
    values <- data[[column]]
    values <- values[!is.na(values)]
    length(unique(values)) >= 2
  }, logical(1)))
}

# Canonical reader ids from exported usernames (p001, p013, ...).
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

clmm_random_effect_sds <- function(model) {
  st <- tryCatch(summary(model)$ST, error = function(error) NULL)
  if (is.null(st) || !is.list(st)) {
    return(numeric())
  }

  unlist(lapply(st, function(component) {
    if (is.matrix(component)) {
      abs(as.numeric(component[1, 1]))
    } else {
      NA_real_
    }
  }), use.names = FALSE)
}

clmm_is_singular_fit <- function(model, tolerance = 1e-2) {
  sds <- clmm_random_effect_sds(model)
  length(sds) > 0 && all(sds < tolerance, na.rm = TRUE)
}

clmm_singular_fit_message <- function() {
  paste(
    "*** SINGULAR FIT ***",
    "All random-effect standard deviations are near zero (variance on the boundary).",
    "With n ~ 60 and ~15 readers, reader / book / src_lang random intercepts are not separately identifiable.",
    "Fixed effects for type (HT vs MT) and order (HT-first vs MT-first) may still be useful; random-effect inference is not.",
    "Consider a simpler random structure, e.g. (1 | reader) only.",
    sep = "\n"
  )
}

ordinal_rating_labels <- c(
  q1 = "Acceptability (1 unacceptable -> 5 acceptable)",
  q2 = "Smoothness (1 unsmooth -> 5 smooth)",
  q3 = "Immersion (1 interfered -> 5 supported)",
  q4 = "Continue reading (1 no -> 5 yes)"
)

write_model_summary <- function(
  model,
  warnings,
  output_path,
  test_label,
  singular_note = NULL,
  question_label = NULL,
  preamble = NULL
) {
  ci_lines <- capture.output({
    ci <- tryCatch(
      suppressMessages(confint(model, method = "Wald")),
      error = function(error) error
    )
    print(ci)
  })

  warning_lines <- if (length(warnings) == 0) {
    "No warnings captured."
  } else {
    paste0("* ", warnings, collapse = "\n")
  }

  preamble_lines <- if (is.null(preamble)) {
    character()
  } else {
    c(preamble, "")
  }

  label_lines <- if (is.null(question_label)) {
    character()
  } else {
    c("Question", question_label, "")
  }

  singular_lines <- if (is.null(singular_note)) {
    character()
  } else {
    c("Singular fit", singular_note, "")
  }

  output <- c(
    test_label,
    "",
    preamble_lines,
    label_lines,
    singular_lines,
    "Model summary",
    capture.output(print(summary(model))),
    "",
    "Confidence intervals",
    ci_lines,
    "",
    "Model warnings",
    warning_lines
  )

  cat(paste(output, collapse = "\n"), "\n")
  writeLines(output, output_path)
}

write_skip_note <- function(output_path, response, reason, test_label) {
  output <- c(test_label, "", paste("Skipped model for", response), reason)
  cat(paste(output, collapse = "\n"), "\n")
  writeLines(output, output_path)
}

write_preference_model_output <- function(
  model,
  output_path,
  test_label,
  warnings = character(),
  preamble = NULL,
  emmeans_specs = list(),
  singular_check = FALSE
) {
  ci_lines <- capture.output({
    ci <- tryCatch(
      suppressMessages(confint(model, method = "Wald")),
      error = function(error) error
    )
    print(ci)
  })

  emmeans_lines <- character()
  for (spec in emmeans_specs) {
    emmeans_lines <- c(
      emmeans_lines,
      spec$label,
      capture.output({
        emm <- tryCatch(
          emmeans(model, spec$formula, type = "response"),
          error = function(error) paste("emmeans failed:", conditionMessage(error))
        )
        print(emm)
      }),
      ""
    )
  }

  warning_lines <- if (length(warnings) == 0) {
    "No warnings captured."
  } else {
    paste0("* ", warnings, collapse = "\n")
  }

  preamble_lines <- if (is.null(preamble)) {
    character()
  } else {
    c(preamble, "")
  }

  singular_lines <- character()
  if (singular_check) {
    singular_value <- tryCatch(isSingular(model), error = function(error) NA)
    singular_lines <- c(
      "Singular fit check",
      paste("isSingular(model):", singular_value),
      if (isTRUE(singular_value)) {
        "Singular fit detected. Prefer simpler GLM without random effects."
      } else {
        character()
      },
      ""
    )
  }

  varcorr_lines <- character()
  if (inherits(model, "merMod")) {
    varcorr_lines <- c(
      "Random effects (VarCorr)",
      capture.output(print(VarCorr(model))),
      ""
    )
  }

  output <- c(
    test_label,
    "",
    preamble_lines,
    singular_lines,
    "Model summary",
    capture.output(print(summary(model))),
    "",
    varcorr_lines,
    "Confidence intervals",
    ci_lines,
    if (length(emmeans_lines) > 0) c("Estimated marginal means", emmeans_lines) else character(),
    "Model warnings",
    warning_lines
  )

  cat(paste(output, collapse = "\n"), "\n")
  writeLines(output, output_path)
}

fit_preference_model <- function(
  formula,
  data,
  output_path,
  test_label,
  family = binomial,
  emmeans_specs = list(),
  singular_check = FALSE
) {
  model_data <- data %>% filter(complete.cases(.))

  if (nrow(model_data) < 4) {
    write_skip_note(output_path, basename(output_path), "Not enough complete cases for model fitting.", test_label)
    return(NULL)
  }

  fitted <- tryCatch(
    capture_model_warnings(
      if (identical(family, binomial)) {
        glm(formula, family = binomial, data = model_data)
      } else if (identical(family, quasibinomial)) {
        glm(formula, family = quasibinomial, data = model_data)
      } else {
        glmer(
          formula,
          data = model_data,
          family = binomial,
          control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
        )
      }
    ),
    error = function(error) {
      write_skip_note(output_path, basename(output_path), paste("Model failed:", conditionMessage(error)), test_label)
      NULL
    }
  )
  if (is.null(fitted)) return(NULL)

  preamble <- paste(
    "Formula:",
    paste(deparse(formula), collapse = " "),
    sep = "\n"
  )

  write_preference_model_output(
    fitted$value,
    output_path,
    test_label = test_label,
    warnings = fitted$warnings,
    preamble = preamble,
    emmeans_specs = emmeans_specs,
    singular_check = singular_check
  )
  invisible(fitted$value)
}

write_clm_summary <- function(
  model,
  response,
  warnings,
  output_path,
  test_label,
  question_label = NULL,
  preamble = NULL,
  fixed_effect = "type"
) {
  ci_lines <- capture.output({
    ci <- tryCatch(
      suppressMessages(confint(model, method = "Wald")),
      error = function(error) error
    )
    print(ci)
  })

  emmeans_mean_class <- capture.output(
    print(emmeans(model, as.formula(paste("~", fixed_effect)), mode = "mean.class"))
  )
  emmeans_by_level <- capture.output(
    print(emmeans(model, as.formula(paste("~", fixed_effect, "|", response)), mode = "prob"))
  )

  warning_lines <- if (length(warnings) == 0) {
    "No warnings captured."
  } else {
    paste0("* ", warnings, collapse = "\n")
  }

  preamble_lines <- if (is.null(preamble)) {
    character()
  } else {
    c(preamble, "")
  }

  label_lines <- if (is.null(question_label)) {
    character()
  } else {
    c("Question", question_label, "")
  }

  output <- c(
    test_label,
    "",
    preamble_lines,
    label_lines,
    "Model summary",
    capture.output(print(summary(model))),
    "",
    "Confidence intervals",
    ci_lines,
    "",
    paste0("Estimated marginal means (mean.class by ", fixed_effect, ")"),
    emmeans_mean_class,
    "",
    paste0("Estimated marginal means (prob by ", fixed_effect, " | ", response, ")"),
    emmeans_by_level,
    "",
    "Model warnings",
    warning_lines
  )

  cat(paste(output, collapse = "\n"), "\n")
  writeLines(output, output_path)
}

write_ai_conf_clm_summary <- function(
  model,
  warnings,
  output_path,
  test_label,
  emmeans_formula,
  emmeans_label,
  question_label = NULL,
  preamble = NULL
) {
  ci_lines <- capture.output({
    ci <- tryCatch(
      suppressMessages(confint(model, method = "Wald")),
      error = function(error) error
    )
    print(ci)
  })

  emmeans_lines <- capture.output(
    print(emmeans(model, emmeans_formula, mode = "mean.class"))
  )

  warning_lines <- if (length(warnings) == 0) {
    "No warnings captured."
  } else {
    paste0("* ", warnings, collapse = "\n")
  }

  preamble_lines <- if (is.null(preamble)) {
    character()
  } else {
    c(preamble, "")
  }

  label_lines <- if (is.null(question_label)) {
    character()
  } else {
    c("Question", question_label, "")
  }

  output <- c(
    test_label,
    "",
    preamble_lines,
    label_lines,
    "Model summary",
    capture.output(print(summary(model))),
    "",
    "Confidence intervals",
    ci_lines,
    "",
    emmeans_label,
    emmeans_lines,
    "",
    "Model warnings",
    warning_lines
  )

  cat(paste(output, collapse = "\n"), "\n")
  writeLines(output, output_path)
}

fit_ai_confidence_clm <- function(
  data,
  formula,
  output_path,
  test_label,
  emmeans_formula,
  question_label = NULL,
  emmeans_label = NULL
) {
  if (is.null(emmeans_label)) {
    emmeans_label <- paste0(
      "emmeans(model, ",
      deparse(emmeans_formula),
      ', mode = "mean.class")'
    )
  }
  model_data <- data %>% filter(complete.cases(.))

  if (nrow(model_data) < 4) {
    write_skip_note(output_path, basename(output_path), "Not enough complete cases for CLM.", test_label)
    return(NULL)
  }

  fitted <- tryCatch(
    capture_model_warnings(clm(formula, data = model_data, Hess = TRUE)),
    error = function(error) {
      write_skip_note(output_path, basename(output_path), paste("clm failed:", conditionMessage(error)), test_label)
      NULL
    }
  )
  if (is.null(fitted)) return(NULL)

  model_note <- paste(
    "Formula:",
    paste(deparse(formula), collapse = " "),
    sep = "\n"
  )

  write_ai_conf_clm_summary(
    fitted$value,
    fitted$warnings,
    output_path,
    test_label = test_label,
    emmeans_formula = emmeans_formula,
    emmeans_label = emmeans_label,
    question_label = question_label,
    preamble = model_note
  )
  fitted$value
}

write_capture_output <- function(path, test_label, ...) {
  lines <- c(test_label, "", capture.output(...))
  cat(paste(lines, collapse = "\n"), "\n")
  writeLines(lines, path)
}

fit_lmer_response <- function(data, response, group, output_path, test_label) {
  model_data <- data %>%
    select(all_of(c(response, group, "order", "reader", "book"))) %>%
    filter(complete.cases(.))

  required <- c(response, group, "order", "reader", "book")
  if (nrow(model_data) < 4 || !has_enough_levels(model_data, required)) {
    write_skip_note(output_path, response, "Not enough rows or response/group/order/random-effect levels for lmer.", test_label)
    return(NULL)
  }

  formula <- as.formula(paste(response, "~", group, "+ order + (1 | reader) + (1 | book)"))
  fitted <- tryCatch(
    capture_model_warnings(
      lmer(formula, data = model_data, control = lmerControl(optimizer = "bobyqa"))
    ),
    error = function(error) {
      write_skip_note(output_path, response, paste("lmer failed:", conditionMessage(error)), test_label)
      NULL
    }
  )
  if (is.null(fitted)) return(NULL)
  write_model_summary(fitted$value, fitted$warnings, output_path, test_label = test_label)
  fitted$value
}

fit_glmer_response <- function(data, response, group, output_path, test_label) {
  model_data <- data %>%
    select(all_of(c(response, group, "order", "reader", "book"))) %>%
    filter(complete.cases(.))

  required <- c(response, group, "order", "reader", "book")
  if (nrow(model_data) < 4 || !has_enough_levels(model_data, required)) {
    write_skip_note(output_path, response, "Not enough rows or response/group/order/random-effect levels for glmer.", test_label)
    return(NULL)
  }

  model_data[[response]] <- droplevels(factor(model_data[[response]]))
  if (nlevels(model_data[[response]]) != 2) {
    write_skip_note(output_path, response, "glmer binomial requires exactly two response levels.", test_label)
    return(NULL)
  }

  formula <- as.formula(paste(response, "~", group, "+ order + (1 | reader) + (1 | book)"))
  fitted <- tryCatch(
    capture_model_warnings(
      glmer(
        formula,
        data = model_data,
        family = binomial,
        control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
      )
    ),
    error = function(error) {
      write_skip_note(output_path, response, paste("glmer failed:", conditionMessage(error)), test_label)
      NULL
    }
  )
  if (is.null(fitted)) return(NULL)
  write_model_summary(fitted$value, fitted$warnings, output_path, test_label = test_label)
  fitted$value
}

fit_clmm_response <- function(
  data,
  response,
  output_path,
  test_label,
  question_label = NULL
) {
  model_data <- data %>%
    select(all_of(c(response, "type", "order", "src_lang", "book", "reader"))) %>%
    filter(complete.cases(.))

  required <- c(response, "type", "order", "src_lang", "book", "reader")
  if (nrow(model_data) < 4 || !has_enough_levels(model_data, required)) {
    write_skip_note(output_path, response, "Not enough rows or response/type/order/src_lang/book/reader levels for clmm.", test_label)
    return(NULL)
  }

  formula <- as.formula(
    paste(response, "~ type + order + (1 | reader) + (1 | book)")
  )

  fitted <- tryCatch(
    capture_model_warnings(clmm(formula, data = model_data, Hess = TRUE)),
    error = function(error) {
      write_skip_note(output_path, response, paste("clmm failed:", conditionMessage(error)), test_label)
      NULL
    }
  )
  if (is.null(fitted)) return(NULL)

  model_note <- paste(
    "Model note",
    paste("Formula:", paste(deparse(formula), collapse = " ")),
    "Fixed effects: type (HT vs MT) and order (HT-first vs MT-first).",
    "Random intercepts: reader and book nested within src_lang.",
    sep = "\n"
  )
  cat("\n", model_note, "\n\n", sep = "")

  singular_note <- NULL
  singular_warning <- any(grepl("singular|boundary", fitted$warnings, ignore.case = TRUE))
  if (singular_warning || clmm_is_singular_fit(fitted$value)) {
    singular_note <- clmm_singular_fit_message()
    cat("\n", singular_note, "\n", sep = "")
  }

  write_model_summary(
    fitted$value,
    fitted$warnings,
    output_path,
    test_label = test_label,
    singular_note = singular_note,
    question_label = question_label,
    preamble = model_note
  )
  fitted$value
}

fit_clm_response <- function(
  data,
  response,
  output_path,
  test_label,
  question_label = NULL
) {
  model_data <- data %>%
    select(all_of(c(response, "type", "order"))) %>%
    filter(complete.cases(.))

  required <- c(response, "type", "order")
  if (nrow(model_data) < 4 || !has_enough_levels(model_data, required)) {
    write_skip_note(output_path, response, "Not enough rows or response/type/order levels for clm.", test_label)
    return(NULL)
  }

  formula <- as.formula(paste(response, "~ type + order"))

  fitted <- tryCatch(
    capture_model_warnings(clm(formula, data = model_data, Hess = TRUE)),
    error = function(error) {
      write_skip_note(output_path, response, paste("clm failed:", conditionMessage(error)), test_label)
      NULL
    }
  )
  if (is.null(fitted)) return(NULL)

  model_note <- paste(
    "Model note",
    paste("Formula:", paste(deparse(formula), collapse = " ")),
    "Fixed effects: type (HT vs MT) and order (HT-first vs MT-first).",
    "No random effects (cumulative link model).",
    sep = "\n"
  )
  cat("\n", model_note, "\n\n", sep = "")

  write_clm_summary(
    fitted$value,
    response,
    fitted$warnings,
    output_path,
    test_label = test_label,
    question_label = question_label,
    preamble = model_note
  )
  fitted$value
}

comparison_ordinal_labels <- c(
  preferred_overall = "Overall preference (MT < NO DIFF < HT)",
  smoother = "Smoother translation (MT < NO DIFF < HT)"
)

part2_ordinal_labels <- c(
  difficulty = "Relative translation quality (similar < better < significantly better)"
)

fit_clmm_group_response <- function(
  data,
  response,
  output_path,
  test_label,
  question_label = NULL,
  group_col = "group"
) {
  model_data <- data %>%
    select(all_of(c(response, group_col, "order", "reader", "book"))) %>%
    filter(complete.cases(.))

  required <- c(response, group_col, "order", "reader", "book")
  if (nrow(model_data) < 4 || !has_enough_levels(model_data, required)) {
    write_skip_note(
      output_path,
      response,
      "Not enough rows or response/group/order/reader/book levels for clmm.",
      test_label
    )
    return(NULL)
  }

  formula <- as.formula(
    paste(response, "~", group_col, "+ order + (1 | reader) + (1 | book)")
  )

  fitted <- tryCatch(
    capture_model_warnings(clmm(formula, data = model_data, Hess = TRUE)),
    error = function(error) {
      write_skip_note(output_path, response, paste("clmm failed:", conditionMessage(error)), test_label)
      NULL
    }
  )
  if (is.null(fitted)) return(NULL)

  model_note <- paste(
    "Model note",
    paste("Formula:", paste(deparse(formula), collapse = " ")),
    paste0(
      "Fixed effects: ", group_col,
      " (source language) and order (HT-first vs MT-first)."
    ),
    "Random intercepts: reader and book.",
    "Response is an ordered factor (cumulative link mixed model).",
    sep = "\n"
  )
  cat("\n", model_note, "\n\n", sep = "")

  singular_note <- NULL
  singular_warning <- any(grepl("singular|boundary", fitted$warnings, ignore.case = TRUE))
  if (singular_warning || clmm_is_singular_fit(fitted$value)) {
    singular_note <- clmm_singular_fit_message()
    cat("\n", singular_note, "\n", sep = "")
  }

  write_model_summary(
    fitted$value,
    fitted$warnings,
    output_path,
    test_label = test_label,
    singular_note = singular_note,
    question_label = question_label,
    preamble = model_note
  )
  fitted$value
}

fit_clm_group_response <- function(
  data,
  response,
  output_path,
  test_label,
  question_label = NULL,
  group_col = "group"
) {
  model_data <- data %>%
    select(all_of(c(response, group_col, "order"))) %>%
    filter(complete.cases(.))

  required <- c(response, group_col, "order")
  if (nrow(model_data) < 4 || !has_enough_levels(model_data, required)) {
    write_skip_note(
      output_path,
      response,
      "Not enough rows or response/group/order levels for clm.",
      test_label
    )
    return(NULL)
  }

  formula <- as.formula(paste(response, "~", group_col, "+ order"))

  fitted <- tryCatch(
    capture_model_warnings(clm(formula, data = model_data, Hess = TRUE)),
    error = function(error) {
      write_skip_note(output_path, response, paste("clm failed:", conditionMessage(error)), test_label)
      NULL
    }
  )
  if (is.null(fitted)) return(NULL)

  model_note <- paste(
    "Model note",
    paste("Formula:", paste(deparse(formula), collapse = " ")),
    paste0(
      "Fixed effects: ", group_col,
      " (source language) and order (HT-first vs MT-first)."
    ),
    "No random effects (cumulative link model).",
    sep = "\n"
  )
  cat("\n", model_note, "\n\n", sep = "")

  write_clm_summary(
    fitted$value,
    response,
    fitted$warnings,
    output_path,
    test_label = test_label,
    question_label = question_label,
    preamble = model_note,
    fixed_effect = group_col
  )
  fitted$value
}


########### CLEANING AND VARIABLE TYPES ####################

if (!"order" %in% names(part1)) {
  part1$order <- ifelse(part1$first_version == "HT", "HT-first", "MT-first")
}

part1 <- part1 %>%
  mutate(
    reader = factor(user_id),
    book = factor(book_id),
    source_lang = factor(source_lang),
    order = factor(order, levels = c("HT-first", "MT-first")),
    first_version = factor(first_version),
    second_version = factor(second_version)
  )

if (!"order" %in% names(part2)) {
  part2$order <- ""
}

part2 <- part2 %>%
  mutate(
    reader = factor(user_id),
    book = factor(book_id),
    source_lang = factor(source_lang),
    order = factor(order, levels = c("HT-first", "MT-first")),
    preferred_translation = factor(preferred_translation),
    difficulty = factor(difficulty),
    chunk_id = factor(chunk_id)
  )

part2_span <- part2_span %>%
  mutate(
    reader = factor(user_id),
    book = factor(book_id),
    source_lang = factor(source_lang),
    preferred_translation = factor(preferred_translation),
    version = factor(version),
    label = factor(label),
    chunk_id = factor(chunk_id)
  )

ordinal_question_columns <- c(
  "first_q1", "first_q2", "first_q3", "first_q4",
  "second_q1", "second_q2", "second_q3", "second_q4"
)

word_count_columns <- c(
  "first_q5_nbr_words", "first_q6_nbr_words",
  "second_q5_nbr_words", "second_q6_nbr_words",
  "comparison_q4_nbr_words", "comparison_q7_nbr_words"
)

nominal_question_columns <- c(
  "first_q7_decipher", "second_q7_decipher",
  "comparison_q1_decipher", "comparison_q2_decipher",
  "comparison_q3_decipher", "comparison_q5_decipher"
)

part1[ordinal_question_columns] <- lapply(part1[ordinal_question_columns], as.numeric)
part1[word_count_columns] <- lapply(part1[word_count_columns], as.numeric)
part1[nominal_question_columns] <- lapply(part1[nominal_question_columns], factor)
part1$comparison_q3 <- as.integer(part1$comparison_q3)


########### BUILD ANALYSIS TABLES ####################

# Excerpt- and chunk-level preference tables (used in _shared/ and part_2/).
excerpt_df <- part1 %>%
  transmute(
    reader,
    book,
    order,
    excerpt_preference = comparison_q3_decipher,
    excerpt_strength = case_when(
      comparison_q3 %in% c(1L, 4L) ~ "clear",
      comparison_q3 %in% c(2L, 3L) ~ "slight",
      TRUE ~ NA_character_
    )
  ) %>%
  filter(!is.na(excerpt_preference), excerpt_preference != "") %>%
  mutate(
    excerpt_preference = factor(excerpt_preference, levels = c("MT", "HT")),
    excerpt_pref_HT = as.numeric(excerpt_preference == "HT"),
    excerpt_strength = factor(excerpt_strength, levels = c("slight", "clear")),
    excerpt_clear = as.numeric(excerpt_strength == "clear"),
    order = factor(order, levels = c("HT-first", "MT-first"))
  )

comparison_pref_confidence_df <- excerpt_df %>%
  left_join(
    part1 %>%
      transmute(
        reader,
        book,
        comparison_q3 = as.integer(comparison_q3),
        ai_confidence = as.integer(comparison_q6)
      ),
    by = c("reader", "book")
  ) %>%
  filter(
    !is.na(excerpt_strength),
    !is.na(ai_confidence),
    ai_confidence >= 1L,
    ai_confidence <= 5L
  ) %>%
  mutate(
    confidence_ord = ordered(ai_confidence, levels = 1:5),
    excerpt_strength_factor = factor(excerpt_strength, levels = c("slight", "clear")),
    preference_strength_num = as.integer(excerpt_strength == "clear")
  )

chunk_df <- part2 %>%
  mutate(
    assignment_id = interaction(reader, book, drop = TRUE),
    chunk_uid = paste(book, chunk_id, sep = "::"),
    chunk_index = ave(as.numeric(chunk_id), assignment_id, FUN = rank),
    chunk_index_z = as.numeric(scale(chunk_index)),
    chunk_preference = preferred_translation,
    chunk_strength = recode(
      as.character(difficulty),
      similar_quality = "similar",
      better = "better",
      significantly_better = "significantly_better"
    ),
    chunk_pref_HT = as.numeric(as.character(preferred_translation) == "HT")
  ) %>%
  mutate(
    chunk_preference = factor(chunk_preference, levels = c("MT", "HT")),
    chunk_strength = ordered(
      chunk_strength,
      levels = c("similar", "better", "significantly_better")
    ),
    order = factor(order, levels = c("HT-first", "MT-first"))
  )

chunk_by_assignment <- chunk_df %>%
  filter(!is.na(chunk_pref_HT)) %>%
  group_by(assignment_id, reader, book, order) %>%
  summarise(
    n_chunks = n(),
    n_HT_chunks = sum(chunk_pref_HT == 1),
    n_MT_chunks = sum(chunk_pref_HT == 0),
    prop_HT_chunks = mean(chunk_pref_HT),
    .groups = "drop"
  ) %>%
  mutate(
    chunk_majority = case_when(
      prop_HT_chunks > 0.5 ~ "HT_majority",
      prop_HT_chunks < 0.5 ~ "MT_majority",
      TRUE ~ "tie"
    )
  )

# Single-reading rows compare participant responses by the translation group they saw.
single_reading <- bind_rows(
  part1 %>%
    transmute(
      reader, book, source_lang, order,
      version = first_version,
      q1 = first_q1,
      q2 = first_q2,
      q3 = first_q3,
      q4 = first_q4,
      q5_nbr_words = first_q5_nbr_words,
      q6_nbr_words = first_q6_nbr_words,
      origin_guess = first_q7_decipher
    ),
  part1 %>%
    transmute(
      reader, book, source_lang, order,
      version = second_version,
      q1 = second_q1,
      q2 = second_q2,
      q3 = second_q3,
      q4 = second_q4,
      q5_nbr_words = second_q5_nbr_words,
      q6_nbr_words = second_q6_nbr_words,
      origin_guess = second_q7_decipher
    )
) %>%
  mutate(
    version = factor(version),
    origin_guess = factor(origin_guess)
  )

# Table for ordinal clmm: q ~ type + order + (1 | reader) + (1 | book)
# reader is canonical (collapsed); type is HT/MT; order is HT-first/MT-first; src_lang is fr/ja/pl.
ordinal_clmm_data <- single_reading %>%
  transmute(
    reader = factor(normalize_reader_id(as.character(reader))),
    src_lang = factor(
      ifelse(source_lang == "French", "fr",
        ifelse(source_lang == "Japanese", "ja",
          ifelse(source_lang == "Polish", "pl", as.character(source_lang))
        )
      ),
      levels = c("fr", "ja", "pl")
    ),
    book = factor(sub("^(french|japanese|polish)_eval_", "", tolower(as.character(book)))),
    type = factor(version, levels = c("HT", "MT")),
    order,
    q1 = ordered(as.integer(q1), levels = 1:5),
    q2 = ordered(as.integer(q2), levels = 1:5),
    q3 = ordered(as.integer(q3), levels = 1:5),
    q4 = ordered(as.integer(q4), levels = 1:5)
  ) %>%
  filter(!is.na(type))

# Comparison rows do not have a single exposed HT/MT condition, so source language is used
# as GROUP for mixed models over comparison-question responses.
comparison <- part1 %>%
  transmute(
    reader, book, order,
    group = source_lang,
    preferred_overall = comparison_q1_decipher,
    smoother = comparison_q2_decipher,
    comparison_q4_nbr_words = comparison_q4_nbr_words,
    comparison_q7_nbr_words = comparison_q7_nbr_words
  ) %>%
  mutate(
    preferred_overall = ordered(
      as.character(preferred_overall),
      levels = c("MT", "NO DIFF", "HT")
    ),
    smoother = ordered(
      as.character(smoother),
      levels = c("MT", "NO DIFF", "HT")
    )
  )

part2_analysis <- part2 %>%
  transmute(
    reader, book, order,
    group = source_lang,
    preferred_translation = preferred_translation,
    difficulty = difficulty,
    justification_nbr_words = justification_nbr_words
  ) %>%
  mutate(
    difficulty = ordered(
      as.character(difficulty),
      levels = c("similar_quality", "better", "significantly_better")
    )
  )


########### _SHARED ####################

# _shared/data_structure.txt
sink(OUTPUT_PATHS$shared_data_structure)
cat(TEST_LABELS$descriptive, "\n\n", sep = "")
cat("Part 1 structure\n")
str(part1)
cat("\nPart 1 summary\n")
print(summary(part1))
cat("\nPart 2 structure\n")
str(part2)
cat("\nPart 2 summary\n")
print(summary(part2))
cat("\nPart 2 span structure\n")
str(part2_span)
cat("\nPart 2 span summary\n")
print(summary(part2_span))
sink()

# _shared/preference_descriptives.txt
sink(OUTPUT_PATHS$shared_preference_descriptives)
cat(TEST_LABELS$descriptive, "\n\n", sep = "")
cat("Excerpt- and chunk-level preference descriptives\n\n")
cat("excerpt_df rows:", nrow(excerpt_df), "\n")
cat("chunk_df rows:", nrow(chunk_df), "\n")
cat("chunk_by_assignment rows:", nrow(chunk_by_assignment), "\n\n")
cat("chunk_preference counts\n")
print(table(chunk_df$chunk_preference))
cat("\nchunk_preference proportions\n")
print(prop.table(table(chunk_df$chunk_preference)))
cat("\nchunk_majority counts\n")
print(table(chunk_by_assignment$chunk_majority))
cat("\nprop_HT_chunks summary\n")
print(summary(chunk_by_assignment$prop_HT_chunks))
cat("\nexcerpt_df structure\n")
str(excerpt_df)
cat("\nchunk_df structure\n")
str(chunk_df)
sink()

# _shared/response_variables.txt
sink(OUTPUT_PATHS$shared_response_variables)
cat(TEST_LABELS$descriptive, "\n\n", sep = "")
cat("Participant response variables and intended model types\n\n")
cat("Single-reading ordinal ratings, clmm:\n")
for (response in names(ordinal_rating_labels)) {
  cat(" ", response, " — ", ordinal_rating_labels[[response]], "\n", sep = "")
}
cat("  formula on ordinal_clmm_data: q* ~ type + order + (1 | reader) + (1 | book)\n")
cat("Single-reading ordinal q2/q3 CLM fallback when CLMM is singular: q* ~ type + order\n")
cat("Single-reading continuous word counts, lmer: q5_nbr_words, q6_nbr_words\n")
cat("Single-reading nominal/binary origin guess, glmer when binary: origin_guess\n")
cat("Part 1 comparison ordinal answers, clmm: preferred_overall, smoother (MT < NO DIFF < HT)\n")
cat("Part 1 comparison ordinal CLM fallback when CLMM is singular: preferred_overall, smoother\n")
cat("AI identification (single_ai, comparison_ai): see part_1/single_reading/ai_identification/, part_1/comparison/ai_identification/\n")
cat("Part 1 comparison continuous word counts, lmer: comparison_q4_nbr_words, comparison_q7_nbr_words\n")
cat("Part 1 comparison preference strength vs AI confidence (Q3 vs Q6), correlation/clm/glm\n")
cat("  see part_1/comparison/preference_confidence/\n")
cat("Part 1 comparison perceived HT preference, binomial/glm/glmer: prefers_perceived_HT\n")
cat("  (Q3 continue-reading preference vs Q5 AI identification; see part_1/comparison/perceived_ht_preference/)\n")
cat("Part 2 binary chunk preference, glmer: preferred_translation\n")
cat("Part 2 ordinal difficulty, clmm: difficulty (similar_quality < better < significantly_better)\n")
cat("Part 2 ordinal difficulty CLM fallback when CLMM is singular: difficulty\n")
cat("Part 2 continuous justification length, lmer: justification_nbr_words\n\n")
cat("Single-reading table\n")
str(single_reading)
cat("\nOrdinal clmm table\n")
str(ordinal_clmm_data)
cat("\nComparison table\n")
str(comparison)
cat("\nPart 2 analysis table\n")
str(part2_analysis)
sink()


########### PART 1 ####################

# AI identification tables (single-reading and comparison).
single_ai <- bind_rows(
  part1 %>%
    transmute(
      person_id = normalize_reader_id(participant_id),
      book_id = book_id,
      order = order,
      single_read_position = 1L,
      actual_version = as.character(first_version),
      guess_version = as.character(first_q7_decipher),
      confidence_score = as.integer(first_q8)
    ),
  part1 %>%
    transmute(
      person_id = normalize_reader_id(participant_id),
      book_id = book_id,
      order = order,
      single_read_position = 2L,
      actual_version = as.character(second_version),
      guess_version = as.character(second_q7_decipher),
      confidence_score = as.integer(second_q8)
    )
) %>%
  mutate(
    assignment_id = interaction(person_id, book_id, drop = TRUE),
    person_id = factor(person_id),
    book_id = factor(book_id),
    order = factor(order, levels = c("HT-first", "MT-first")),
    actual_version = factor(actual_version, levels = c("HT", "MT")),
    guess_version = factor(guess_version, levels = c("HT", "MT")),
    single_read_position = factor(single_read_position, levels = c(1, 2)),
    actual_MT = as.numeric(actual_version == "MT"),
    guessed_MT = as.numeric(guess_version == "MT"),
    correct_guess = as.numeric(guess_version == actual_version)
  )

comparison_ai <- part1 %>%
  transmute(
    person_id = normalize_reader_id(participant_id),
    book_id = book_id,
    order = order,
    ai_guess_version = as.character(comparison_q5_decipher),
    confidence_score = as.integer(comparison_q6)
  ) %>%
  mutate(
    assignment_id = interaction(person_id, book_id, drop = TRUE),
    person_id = factor(person_id),
    book_id = factor(book_id),
    order = factor(order, levels = c("HT-first", "MT-first")),
    ai_guess_version = factor(ai_guess_version, levels = c("HT", "MT")),
    ai_guess_correct = as.numeric(ai_guess_version == "MT")
  )

comparison_pref_perception_df <- excerpt_df %>%
  left_join(
    comparison_ai %>%
      transmute(
        reader = person_id,
        book = book_id,
        ai_guess_version
      ),
    by = c("reader", "book")
  ) %>%
  filter(
    !is.na(excerpt_preference),
    !is.na(ai_guess_version)
  ) %>%
  mutate(
    perceived_HT_version = factor(
      ifelse(ai_guess_version == "HT", "MT", "HT"),
      levels = c("MT", "HT")
    ),
    prefers_perceived_HT = as.numeric(
      excerpt_preference == perceived_HT_version
    ),
    prefers_actual_HT = excerpt_pref_HT,
    ai_guess_correct = as.numeric(ai_guess_version == "MT")
  )

single_ai_guess <- single_ai %>%
  filter(!is.na(actual_version), !is.na(guess_version))

conf_single <- table(
  actual = single_ai_guess$actual_version,
  guessed = single_ai_guess$guess_version
)

true_HT_guessed_HT <- conf_single["HT", "HT"]
true_HT_guessed_MT <- conf_single["HT", "MT"]
true_MT_guessed_HT <- conf_single["MT", "HT"]
true_MT_guessed_MT <- conf_single["MT", "MT"]

single_accuracy <- (
  true_HT_guessed_HT + true_MT_guessed_MT
) / sum(conf_single)

single_sensitivity_MT <- true_MT_guessed_MT / sum(conf_single["MT", ])
single_specificity_HT <- true_HT_guessed_HT / sum(conf_single["HT", ])
single_false_positive_MT <- true_HT_guessed_MT / sum(conf_single["HT", ])
single_bias_guess_MT <- sum(conf_single[, "MT"]) / sum(conf_single)

single_identification_summary <- data.frame(
  accuracy = single_accuracy,
  sensitivity_MT = single_sensitivity_MT,
  specificity_HT = single_specificity_HT,
  false_positive_MT = single_false_positive_MT,
  bias_guess_MT = single_bias_guess_MT
)


########### part_1/single_reading/ai_identification/ ####################

write_capture_output(
  OUTPUT_PATHS$p1_single_ai_confusion,
  TEST_LABELS$descriptive,
  cat("Confusion matrix (actual x guessed)\n"),
  print(conf_single),
  cat("\nRow proportions\n"),
  print(prop.table(conf_single, margin = 1)),
  cat("\nCell counts\n"),
  cat("true_HT_guessed_HT:", true_HT_guessed_HT, "\n"),
  cat("true_HT_guessed_MT:", true_HT_guessed_MT, "\n"),
  cat("true_MT_guessed_HT:", true_MT_guessed_HT, "\n"),
  cat("true_MT_guessed_MT:", true_MT_guessed_MT, "\n"),
  cat("\nIdentification summary\n"),
  print(single_identification_summary)
)

single_ai_model <- single_ai_guess %>%
  filter(!is.na(guessed_MT), !is.na(actual_version), !is.na(single_read_position))

cat("\n---------- guessed_MT_glm_binomial ----------\n")
invisible(fit_preference_model(
  guessed_MT ~ actual_version + single_read_position,
  single_ai_model,
  OUTPUT_PATHS$p1_single_ai_guessed_mt_glm,
  test_label = TEST_LABELS$glm_binomial,
  family = binomial,
  emmeans_specs = list(
    list(
      formula = ~actual_version,
      label = "emmeans(model, ~ actual_version, type = \"response\")"
    )
  )
))

cat("\n---------- guessed_MT_glmer_binomial ----------\n")
invisible(fit_preference_model(
  guessed_MT ~ actual_version + single_read_position + (1 | person_id) + (1 | book_id),
  single_ai_model,
  OUTPUT_PATHS$p1_single_ai_guessed_mt_glmer,
  test_label = TEST_LABELS$glmer_binomial,
  family = glmer,
  singular_check = TRUE
))

single_correct <- single_ai %>%
  filter(!is.na(correct_guess))

write_capture_output(
  OUTPUT_PATHS$p1_single_ai_correct_binom,
  TEST_LABELS$binom,
  print(binom.test(
    sum(single_correct$correct_guess == 1, na.rm = TRUE),
    sum(!is.na(single_correct$correct_guess)),
    p = 0.5
  ))
)

single_ai_conf <- single_ai %>%
  filter(!is.na(confidence_score), !is.na(correct_guess), !is.na(actual_version)) %>%
  mutate(
    confidence_ord = ordered(confidence_score, levels = c(1, 2, 3, 4, 5)),
    correct_guess_factor = factor(
      ifelse(correct_guess == 1, "correct", "wrong"),
      levels = c("wrong", "correct")
    )
  )

cat("\n---------- confidence_clm ----------\n")
fit_ai_confidence_clm(
  single_ai_conf,
  confidence_ord ~ correct_guess_factor + actual_version,
  OUTPUT_PATHS$p1_single_ai_confidence_clm,
  test_label = TEST_LABELS$clm,
  emmeans_formula = ~correct_guess_factor,
  question_label = "Single-reading confidence (Q8) by guess correctness and actual version"
)

comparison_ai_model <- comparison_ai %>%
  filter(!is.na(ai_guess_correct), !is.na(order))


########### part_1/comparison/ai_identification/ ####################

write_capture_output(
  OUTPUT_PATHS$p1_comp_ai_descriptives,
  TEST_LABELS$descriptive,
  cat("table(ai_guess_correct)\n"),
  print(table(comparison_ai_model$ai_guess_correct)),
  cat("\nmean(ai_guess_correct)\n"),
  print(mean(comparison_ai_model$ai_guess_correct, na.rm = TRUE))
)

write_capture_output(
  OUTPUT_PATHS$p1_comp_ai_correct_binom,
  TEST_LABELS$binom,
  print(binom.test(
    sum(comparison_ai_model$ai_guess_correct == 1, na.rm = TRUE),
    sum(!is.na(comparison_ai_model$ai_guess_correct)),
    p = 0.5
  ))
)

cat("\n---------- ai_guess_glm_binomial ----------\n")
invisible(fit_preference_model(
  ai_guess_correct ~ order,
  comparison_ai_model,
  OUTPUT_PATHS$p1_comp_ai_guess_glm,
  test_label = TEST_LABELS$glm_binomial,
  family = binomial,
  emmeans_specs = list(
    list(formula = ~order, label = "emmeans(model, ~ order, type = \"response\")"),
    list(formula = ~1, label = "emmeans(model, ~ 1, type = \"response\")")
  )
))

single_stage_assignment <- single_ai %>%
  group_by(assignment_id, person_id, book_id, order) %>%
  summarise(
    n_correct = sum(correct_guess == 1, na.rm = TRUE),
    n_wrong = sum(correct_guess == 0, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(stage = "single_reading")

comparison_stage_assignment <- comparison_ai %>%
  group_by(assignment_id, person_id, book_id, order) %>%
  summarise(
    n_correct = sum(ai_guess_correct == 1, na.rm = TRUE),
    n_wrong = sum(ai_guess_correct == 0, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(stage = "comparison")

identification_by_stage <- bind_rows(
  single_stage_assignment,
  comparison_stage_assignment
) %>%
  mutate(
    stage = factor(stage, levels = c("single_reading", "comparison")),
    order = factor(order, levels = c("HT-first", "MT-first"))
  )

cat("\n---------- stage_accuracy_glm_quasibinomial ----------\n")
invisible(fit_preference_model(
  cbind(n_correct, n_wrong) ~ stage + order,
  identification_by_stage,
  OUTPUT_PATHS$p1_comp_stage_accuracy_glm,
  test_label = TEST_LABELS$glm_quasibinomial,
  family = quasibinomial,
  emmeans_specs = list(
    list(formula = ~stage, label = "emmeans(model, ~ stage, type = \"response\")")
  )
))

comparison_ai_conf <- comparison_ai %>%
  filter(!is.na(confidence_score), !is.na(ai_guess_correct)) %>%
  mutate(
    confidence_ord = ordered(confidence_score, levels = c(1, 2, 3, 4, 5)),
    correct_guess_factor = factor(
      ifelse(ai_guess_correct == 1, "correct", "wrong"),
      levels = c("wrong", "correct")
    )
  )

cat("\n---------- confidence_clm ----------\n")
fit_ai_confidence_clm(
  comparison_ai_conf,
  confidence_ord ~ correct_guess_factor + order,
  OUTPUT_PATHS$p1_comp_ai_confidence_clm,
  test_label = TEST_LABELS$clm,
  emmeans_formula = ~correct_guess_factor,
  question_label = "Comparison confidence (Q6) by MT-identification correctness and order"
)


########### part_1/single_reading/ordinal/ ####################

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

# q* ~ type + order + (1 | reader) + (1 | book)
for (response in names(ordinal_rating_labels)) {
  output_path <- single_clmm_paths[[response]]
  question_label <- ordinal_rating_labels[[response]]
  cat("\n---------- ", basename(output_path), ": ", question_label, " ----------\n", sep = "")
  fit_clmm_response(
    ordinal_clmm_data,
    response,
    output_path,
    test_label = TEST_LABELS$clmm,
    question_label = question_label
  )
}

# CLM fallback when CLMM is singular (q2, q3).
for (response in c("q2", "q3")) {
  output_path <- single_clm_paths[[response]]
  cat("\n---------- ", basename(output_path), ": ", ordinal_rating_labels[[response]], " ----------\n", sep = "")
  fit_clm_response(
    ordinal_clmm_data,
    response,
    output_path,
    test_label = TEST_LABELS$clm,
    question_label = ordinal_rating_labels[[response]]
  )
}


########### part_1/single_reading/word_count/ ####################

fit_lmer_response(
  single_reading,
  "q5_nbr_words",
  "version",
  OUTPUT_PATHS$p1_single_q5_lmer,
  TEST_LABELS$lmer
)
fit_lmer_response(
  single_reading,
  "q6_nbr_words",
  "version",
  OUTPUT_PATHS$p1_single_q6_lmer,
  TEST_LABELS$lmer
)


########### part_1/single_reading/origin_guess/ ####################

fit_glmer_response(
  single_reading,
  "origin_guess",
  "version",
  OUTPUT_PATHS$p1_single_origin_glmer,
  TEST_LABELS$glmer_binomial
)


########### part_1/comparison/ordinal/ ####################

comparison_clmm_paths <- list(
  preferred_overall = OUTPUT_PATHS$p1_comp_preferred_overall_clmm,
  smoother = OUTPUT_PATHS$p1_comp_smoother_clmm
)
comparison_clm_paths <- list(
  preferred_overall = OUTPUT_PATHS$p1_comp_preferred_overall_clm,
  smoother = OUTPUT_PATHS$p1_comp_smoother_clm
)

# 3-level ordinal (MT < NO DIFF < HT); glmer is not applicable.
for (response in names(comparison_ordinal_labels)) {
  output_path <- comparison_clmm_paths[[response]]
  question_label <- comparison_ordinal_labels[[response]]
  cat("\n---------- ", basename(output_path), ": ", question_label, " ----------\n", sep = "")
  fit_clmm_group_response(
    comparison,
    response,
    output_path,
    test_label = TEST_LABELS$clmm,
    question_label = question_label
  )
}

for (response in names(comparison_ordinal_labels)) {
  output_path <- comparison_clm_paths[[response]]
  question_label <- comparison_ordinal_labels[[response]]
  cat("\n---------- ", basename(output_path), ": ", question_label, " ----------\n", sep = "")
  fit_clm_group_response(
    comparison,
    response,
    output_path,
    test_label = TEST_LABELS$clm,
    question_label = question_label
  )
}


########### part_1/comparison/word_count/ ####################

fit_lmer_response(
  comparison,
  "comparison_q4_nbr_words",
  "group",
  OUTPUT_PATHS$p1_comp_q4_lmer,
  TEST_LABELS$lmer
)
fit_lmer_response(
  comparison,
  "comparison_q7_nbr_words",
  "group",
  OUTPUT_PATHS$p1_comp_q7_lmer,
  TEST_LABELS$lmer
)


########### part_1/comparison/excerpt_preference/ ####################

cat("\n---------- prefer_HT_glm_binomial ----------\n")
invisible(fit_preference_model(
  excerpt_pref_HT ~ order,
  excerpt_df,
  OUTPUT_PATHS$p1_comp_excerpt_pref_glm,
  test_label = TEST_LABELS$glm_binomial,
  family = binomial,
  emmeans_specs = list(
    list(formula = ~order, label = "emmeans(model, ~ order, type = \"response\")"),
    list(formula = ~1, label = "emmeans(model, ~ 1, type = \"response\")")
  )
))

cat("\n---------- prefer_HT_glmer_binomial ----------\n")
invisible(fit_preference_model(
  excerpt_pref_HT ~ order + (1 | reader) + (1 | book),
  excerpt_df,
  OUTPUT_PATHS$p1_comp_excerpt_pref_glmer,
  test_label = TEST_LABELS$glmer_binomial,
  family = glmer,
  singular_check = TRUE
))

cat("\n---------- preference_strength_glm_binomial ----------\n")
invisible(fit_preference_model(
  excerpt_clear ~ excerpt_preference + order,
  excerpt_df %>% filter(!is.na(excerpt_clear)),
  OUTPUT_PATHS$p1_comp_excerpt_strength_glm,
  test_label = TEST_LABELS$glm_binomial,
  family = binomial,
  emmeans_specs = list(
    list(
      formula = ~excerpt_preference,
      label = "emmeans(model, ~ excerpt_preference, type = \"response\")"
    )
  )
))


########### part_1/comparison/preference_confidence/ ####################

pref_conf_crosstab <- table(
  preference_strength = comparison_pref_confidence_df$excerpt_strength_factor,
  ai_confidence = comparison_pref_confidence_df$ai_confidence
)

write_capture_output(
  OUTPUT_PATHS$p1_comp_pref_conf_descriptives,
  TEST_LABELS$descriptive,
  cat("Q3 preference strength (slight vs clear) vs Q6 AI-identification confidence\n"),
  cat("comparison_q3: 1/4 = clearly better, 2/3 = slightly better\n\n"),
  cat("n assignments:", nrow(comparison_pref_confidence_df), "\n\n"),
  cat("Mean Q6 confidence by Q3 strength\n"),
  print(
    tapply(
      comparison_pref_confidence_df$ai_confidence,
      comparison_pref_confidence_df$excerpt_strength_factor,
      mean
    )
  ),
  cat("\nCrosstab: preference strength (rows) x Q6 confidence (cols)\n"),
  print(pref_conf_crosstab),
  cat("\nColumn proportions\n"),
  print(prop.table(pref_conf_crosstab, margin = 2))
)

write_capture_output(
  OUTPUT_PATHS$p1_comp_pref_conf_correlation,
  TEST_LABELS$correlation,
  cat("Preference strength (clear=1, slight=0) vs Q6 confidence (1-5)\n\n"),
  cat("Spearman rank correlation\n"),
  print(
    cor.test(
      comparison_pref_confidence_df$ai_confidence,
      comparison_pref_confidence_df$preference_strength_num,
      method = "spearman",
      exact = FALSE
    )
  ),
  cat("\nKendall rank correlation\n"),
  print(
    cor.test(
      comparison_pref_confidence_df$ai_confidence,
      comparison_pref_confidence_df$preference_strength_num,
      method = "kendall",
      exact = FALSE
    )
  ),
  cat("\nPearson correlation (numeric Q6 vs binary clear preference)\n"),
  print(
    cor.test(
      comparison_pref_confidence_df$ai_confidence,
      comparison_pref_confidence_df$preference_strength_num,
      method = "pearson"
    )
  ),
  cat("\nSpearman: raw comparison_q3 (1-4) vs Q6 confidence\n"),
  cat("(includes direction and strength; reference only)\n"),
  print(
    cor.test(
      comparison_pref_confidence_df$ai_confidence,
      comparison_pref_confidence_df$comparison_q3,
      method = "spearman",
      exact = FALSE
    )
  )
)

comparison_pref_conf_clm <- comparison_pref_confidence_df %>%
  mutate(
    excerpt_strength_factor = factor(
      excerpt_strength_factor,
      levels = c("slight", "clear")
    )
  )

cat("\n---------- confidence_by_strength_clm ----------\n")
fit_ai_confidence_clm(
  comparison_pref_conf_clm,
  confidence_ord ~ excerpt_strength_factor + order,
  OUTPUT_PATHS$p1_comp_pref_conf_clm,
  test_label = TEST_LABELS$clm,
  emmeans_formula = ~excerpt_strength_factor,
  question_label = "Q6 confidence by Q3 preference strength (slight vs clear) and order"
)

cat("\n---------- strength_by_confidence_glm_binomial ----------\n")
invisible(fit_preference_model(
  excerpt_clear ~ ai_confidence + order,
  comparison_pref_confidence_df,
  OUTPUT_PATHS$p1_comp_pref_conf_strength_glm,
  test_label = TEST_LABELS$glm_binomial,
  family = binomial,
  emmeans_specs = list(
    list(
      formula = ~ai_confidence,
      label = "emmeans(model, ~ ai_confidence, type = \"response\")"
    )
  )
))


########### part_1/comparison/perceived_ht_preference/ ####################

pref_perception_crosstab <- table(
  preference = comparison_pref_perception_df$excerpt_preference,
  ai_identified = comparison_pref_perception_df$ai_guess_version
)

write_capture_output(
  OUTPUT_PATHS$p1_comp_perceived_ht_descriptives,
  TEST_LABELS$descriptive,
  cat("Prefer translation reader treats as human (perceived HT)\n"),
  cat("perceived_HT_version = excerpt not identified as AI on Q5\n"),
  cat("prefers_perceived_HT = excerpt_preference == perceived_HT_version\n\n"),
  cat("n assignments:", nrow(comparison_pref_perception_df), "\n\n"),
  cat("Proportion prefers_perceived_HT:", mean(comparison_pref_perception_df$prefers_perceived_HT), "\n"),
  cat("Proportion prefers_actual_HT:", mean(comparison_pref_perception_df$prefers_actual_HT), "\n\n"),
  cat("Crosstab: continue-reading preference (rows) x excerpt identified as AI (cols)\n"),
  print(pref_perception_crosstab),
  cat("\nRow proportions (preference)\n"),
  print(prop.table(pref_perception_crosstab, margin = 1)),
  cat("\nprefers_perceived_HT by ai_guess_correct (1 = correctly tagged actual MT as AI)\n"),
  print(
    tapply(
      comparison_pref_perception_df$prefers_perceived_HT,
      comparison_pref_perception_df$ai_guess_correct,
      mean,
      na.rm = TRUE
    )
  ),
  cat("\nCounts by ai_guess_correct\n"),
  print(table(comparison_pref_perception_df$ai_guess_correct))
)

write_capture_output(
  OUTPUT_PATHS$p1_comp_prefer_perceived_ht_binom,
  TEST_LABELS$binom,
  print(binom.test(
    sum(comparison_pref_perception_df$prefers_perceived_HT == 1, na.rm = TRUE),
    sum(!is.na(comparison_pref_perception_df$prefers_perceived_HT)),
    p = 0.5
  ))
)

cat("\n---------- prefer_perceived_HT_glm_binomial ----------\n")
invisible(fit_preference_model(
  prefers_perceived_HT ~ order,
  comparison_pref_perception_df,
  OUTPUT_PATHS$p1_comp_prefer_perceived_ht_glm,
  test_label = TEST_LABELS$glm_binomial,
  family = binomial,
  emmeans_specs = list(
    list(formula = ~order, label = "emmeans(model, ~ order, type = \"response\")"),
    list(formula = ~1, label = "emmeans(model, ~ 1, type = \"response\")")
  )
))

cat("\n---------- prefer_perceived_HT_glmer_binomial ----------\n")
invisible(fit_preference_model(
  prefers_perceived_HT ~ order + (1 | reader) + (1 | book),
  comparison_pref_perception_df,
  OUTPUT_PATHS$p1_comp_prefer_perceived_ht_glmer,
  test_label = TEST_LABELS$glmer_binomial,
  family = glmer,
  singular_check = TRUE
))


########### PART 2 ####################


########### part_2/chunk_preference/ ####################

fit_glmer_response(
  part2_analysis,
  "preferred_translation",
  "group",
  OUTPUT_PATHS$p2_preferred_translation_glmer,
  TEST_LABELS$glmer_binomial
)

cat("\n---------- assignment_prop_HT_glm_quasibinomial ----------\n")
invisible(fit_preference_model(
  cbind(n_HT_chunks, n_MT_chunks) ~ order,
  chunk_by_assignment,
  OUTPUT_PATHS$p2_assignment_prop_ht_glm,
  test_label = TEST_LABELS$glm_quasibinomial,
  family = quasibinomial,
  emmeans_specs = list(
    list(formula = ~order, label = "emmeans(model, ~ order, type = \"response\")"),
    list(formula = ~1, label = "emmeans(model, ~ 1, type = \"response\")")
  )
))

cat("\n---------- chunk_level_glmer_binomial ----------\n")
invisible(fit_preference_model(
  chunk_pref_HT ~ order + chunk_index_z + (1 | assignment_id) + (1 | chunk_uid),
  chunk_df %>% filter(!is.na(chunk_pref_HT)),
  OUTPUT_PATHS$p2_chunk_level_glmer,
  test_label = TEST_LABELS$glmer_binomial,
  family = glmer,
  singular_check = TRUE
))

cat("\n---------- reader_book_glmer_binomial ----------\n")
invisible(fit_preference_model(
  chunk_pref_HT ~ 1 + (1 | reader) + (1 | book),
  chunk_df %>% filter(!is.na(chunk_pref_HT)),
  OUTPUT_PATHS$p2_reader_book_glmer,
  test_label = TEST_LABELS$glmer_binomial,
  family = glmer,
  singular_check = TRUE,
  emmeans_specs = list(
    list(formula = ~1, label = "emmeans(model, ~ 1, type = \"response\")")
  )
))


########### part_2/difficulty/ ####################

for (response in names(part2_ordinal_labels)) {
  question_label <- part2_ordinal_labels[[response]]
  cat("\n---------- difficulty_clmm: ", question_label, " ----------\n", sep = "")
  fit_clmm_group_response(
    part2_analysis,
    response,
    OUTPUT_PATHS$p2_difficulty_clmm,
    test_label = TEST_LABELS$clmm,
    question_label = question_label
  )
}

for (response in names(part2_ordinal_labels)) {
  question_label <- part2_ordinal_labels[[response]]
  cat("\n---------- difficulty_clm: ", question_label, " ----------\n", sep = "")
  fit_clm_group_response(
    part2_analysis,
    response,
    OUTPUT_PATHS$p2_difficulty_clm,
    test_label = TEST_LABELS$clm,
    question_label = question_label
  )
}


########### part_2/word_count/ ####################

fit_lmer_response(
  part2_analysis,
  "justification_nbr_words",
  "group",
  OUTPUT_PATHS$p2_justification_lmer,
  TEST_LABELS$lmer
)

cat("Analysis complete. Outputs written to:", analysis_output_root, "\n")
