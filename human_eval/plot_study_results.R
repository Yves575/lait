# Human evaluation result plots.
#
# Run from the project root:
#   Rscript human_eval/plot_study_results.R
#
# Outputs are written to human_eval/figures/ as PDF and PNG files.

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
})

`%||%` <- function(lhs, rhs) {
  if (is.null(lhs) || length(lhs) == 0) rhs else lhs
}


########### PATHS AND LOADING DATA ####################

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- normalizePath(
  sub("^--file=", "", script_arg[1] %||% "human_eval/plot_study_results.R")
)
script_dir <- dirname(script_path)
data_dir <- file.path(script_dir, "data")
figure_dir <- file.path(script_dir, "figures")
dir.create(figure_dir, showWarnings = FALSE, recursive = TRUE)

part1 <- read.csv(
  file.path(data_dir, "part1-study-data-full.csv"),
  na.strings = c("", "NA"),
  stringsAsFactors = FALSE
)
part2 <- read.csv(
  file.path(data_dir, "part2-study-data-full.csv"),
  na.strings = c("", "NA"),
  stringsAsFactors = FALSE
)
part2_span <- read.csv(
  file.path(data_dir, "part2-span-study-data-full.csv"),
  na.strings = c("", "NA"),
  stringsAsFactors = FALSE
)

if (!"user_id" %in% names(part1)) part1$user_id <- part1$participant_id
if (!"user_id" %in% names(part2)) part2$user_id <- part2$participant_id
if (!"user_id" %in% names(part2_span)) part2_span$user_id <- part2_span$participant_id

language_group_column <- Sys.getenv("LITMT_LANGUAGE_GROUP_COLUMN", "source_lang")
language_group_label <- Sys.getenv("LITMT_LANGUAGE_GROUP_LABEL", "Source language")
language_group_label_lower <- tolower(language_group_label)

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

language_levels <- c("English", "French", "Japanese", "Polish", "Spanish")
language_levels <- language_levels[language_levels %in% unique(c(
  part1$source_lang,
  part2$source_lang,
  part2_span$source_lang
))]


########### PLOT HELPERS ####################

theme_litmt <- function(base_size = 11) {
  theme_minimal(base_size = base_size) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      axis.title = element_text(face = "bold"),
      plot.title = element_text(face = "bold", hjust = 0),
      plot.subtitle = element_text(color = "gray30", lineheight = 1.08),
      legend.title = element_text(face = "bold"),
      strip.text = element_text(face = "bold"),
      plot.margin = margin(10, 12, 10, 12)
    )
}

save_plot <- function(plot, filename, width = 8, height = 5) {
  pdf_path <- file.path(figure_dir, paste0(filename, ".pdf"))
  png_path <- file.path(figure_dir, paste0(filename, ".png"))
  ggsave(pdf_path, plot, width = width, height = height, units = "in", device = cairo_pdf)
  ggsave(png_path, plot, width = width, height = height, units = "in", dpi = 300)
}

percent_label <- function(value) {
  paste0(round(value), "%")
}

translation_colors <- c(
  "HT" = "#0072B2",
  "MT" = "#D55E00",
  "NO DIFF" = "#7A7A7A"
)

preference_difficulty_colors <- c(
  "HT_similar_quality" = "#9ECAE1",
  "HT_better" = "#4292C6",
  "HT_significantly_better" = "#08519C",
  "MT_similar_quality" = "#FDBB84",
  "MT_better" = "#EF6548",
  "MT_significantly_better" = "#A63603"
)

preference_difficulty_labels <- c(
  "HT_similar_quality" = "HT, similar",
  "HT_better" = "HT, better",
  "HT_significantly_better" = "HT, significantly better",
  "MT_similar_quality" = "MT, similar",
  "MT_better" = "MT, better",
  "MT_significantly_better" = "MT, significantly better"
)

confidence_labels <- c(
  "1" = "not at all confident",
  "2" = "slightly confident",
  "3" = "moderately confident",
  "4" = "very confident",
  "5" = "extremely confident"
)

ai_identification_colors <- c(
  "HT_1" = "#FDD0A2",
  "HT_2" = "#FDAE6B",
  "HT_3" = "#F16913",
  "HT_4" = "#D94801",
  "HT_5" = "#8C2D04",
  "MT_1" = "#BFE6D8",
  "MT_2" = "#84CFB7",
  "MT_3" = "#4DB894",
  "MT_4" = "#1B9E77",
  "MT_5" = "#006D4E"
)

ai_likely_labels <- c(
  "HT_1" = paste("HT\n", confidence_labels[["1"]]),
  "HT_2" = paste("HT\n", confidence_labels[["2"]]),
  "HT_3" = paste("HT\n", confidence_labels[["3"]]),
  "HT_4" = paste("HT\n", confidence_labels[["4"]]),
  "HT_5" = paste("HT\n", confidence_labels[["5"]]),
  "MT_1" = paste("MT\n", confidence_labels[["1"]]),
  "MT_2" = paste("MT\n", confidence_labels[["2"]]),
  "MT_3" = paste("MT\n", confidence_labels[["3"]]),
  "MT_4" = paste("MT\n", confidence_labels[["4"]]),
  "MT_5" = paste("MT\n", confidence_labels[["5"]])
)

ai_identification_labels <- c(
  "HT_1" = confidence_labels[["1"]],
  "HT_2" = confidence_labels[["2"]],
  "HT_3" = confidence_labels[["3"]],
  "HT_4" = confidence_labels[["4"]],
  "HT_5" = confidence_labels[["5"]],
  "MT_1" = confidence_labels[["1"]],
  "MT_2" = confidence_labels[["2"]],
  "MT_3" = confidence_labels[["3"]],
  "MT_4" = confidence_labels[["4"]],
  "MT_5" = confidence_labels[["5"]]
)

ai_identification_stack_order <- c(
  "HT_5",
  "HT_4",
  "HT_3",
  "HT_2",
  "HT_1",
  "MT_5",
  "MT_4",
  "MT_3",
  "MT_2",
  "MT_1"
)

choice_heatmap_colors <- c(
  "HT_similar_quality" = "#B7E4C7",
  "HT_better" = "#52B788",
  "HT_significantly_better" = "#1B7F5A",
  "MT_similar_quality" = "#D7BDE2",
  "MT_better" = "#9B59B6",
  "MT_significantly_better" = "#5B2C6F"
)

choice_heatmap_labels <- c(
  "HT_similar_quality" = "HT, similar quality",
  "HT_better" = "HT, better",
  "HT_significantly_better" = "HT, significantly better",
  "MT_similar_quality" = "MT, similar quality",
  "MT_better" = "MT, better",
  "MT_significantly_better" = "MT, significantly better"
)

rating_colors <- c(
  "1" = "#B2182B",
  "2" = "#EF8A62",
  "3" = "#F6E8A6",
  "4" = "#67A9CF",
  "5" = "#2166AC"
)

highlight_colors <- c(
  "good" = "#009E73",
  "poor" = "#CC79A7"
)

origin_accuracy_colors <- c(
  "Correct_1" = "#BFE6D8",
  "Correct_2" = "#84CFB7",
  "Correct_3" = "#4DB894",
  "Correct_4" = "#1B9E77",
  "Correct_5" = "#006D4E",
  "Incorrect_1" = "#FDD0A2",
  "Incorrect_2" = "#FDAE6B",
  "Incorrect_3" = "#F16913",
  "Incorrect_4" = "#D94801",
  "Incorrect_5" = "#8C2D04"
)

origin_accuracy_labels <- c(
  "Correct_1" = confidence_labels[["1"]],
  "Correct_2" = confidence_labels[["2"]],
  "Correct_3" = confidence_labels[["3"]],
  "Correct_4" = confidence_labels[["4"]],
  "Correct_5" = confidence_labels[["5"]],
  "Incorrect_1" = confidence_labels[["1"]],
  "Incorrect_2" = confidence_labels[["2"]],
  "Incorrect_3" = confidence_labels[["3"]],
  "Incorrect_4" = confidence_labels[["4"]],
  "Incorrect_5" = confidence_labels[["5"]]
)

origin_accuracy_stack_order <- c(
  "Correct_5",
  "Correct_4",
  "Correct_3",
  "Correct_2",
  "Correct_1",
  "Incorrect_1",
  "Incorrect_2",
  "Incorrect_3",
  "Incorrect_4",
  "Incorrect_5"
)

continue_strength_colors <- c(
  "HT_clear" = "#08519C",
  "HT_slight" = "#6BAED6",
  "MT_slight" = "#FDBB84",
  "MT_clear" = "#A63603"
)

continue_strength_labels <- c(
  "HT_clear" = "HT, clear",
  "HT_slight" = "HT, slight",
  "MT_slight" = "MT, slight",
  "MT_clear" = "MT, clear"
)

ai_likely_colors <- c(
  "HT_1" = "#FDD0A2",
  "HT_2" = "#FDAE6B",
  "HT_3" = "#F16913",
  "HT_4" = "#D94801",
  "HT_5" = "#8C2D04",
  "MT_1" = "#BFE6D8",
  "MT_2" = "#84CFB7",
  "MT_3" = "#4DB894",
  "MT_4" = "#1B9E77",
  "MT_5" = "#006D4E"
)


########### CLEAN AND RESHAPE ####################

single_reading <- bind_rows(
  part1 %>%
    transmute(
      user_id,
      book_id,
      source_lang,
      version = first_version,
      q1 = first_q1,
      q2 = first_q2,
      q3 = first_q3,
      q4 = first_q4
    ),
  part1 %>%
    transmute(
      user_id,
      book_id,
      source_lang,
      version = second_version,
      q1 = second_q1,
      q2 = second_q2,
      q3 = second_q3,
      q4 = second_q4
    )
) %>%
  filter(!is.na(version))

single_long <- bind_rows(
  single_reading %>% transmute(version, question = "Acceptability\n1 unacceptable -> 5 acceptable", rating = q1),
  single_reading %>% transmute(version, question = "Smoothness\n1 unsmooth -> 5 smooth", rating = q2),
  single_reading %>% transmute(version, question = "Immersion\n1 interfered -> 5 supported", rating = q3),
  single_reading %>% transmute(version, question = "Continue reading\n1 no -> 5 yes", rating = q4)
) %>%
  filter(!is.na(rating)) %>%
  mutate(
    version = factor(version, levels = c("HT", "MT")),
    question = factor(
      question,
      levels = c(
        "Acceptability\n1 unacceptable -> 5 acceptable",
        "Smoothness\n1 unsmooth -> 5 smooth",
        "Immersion\n1 interfered -> 5 supported",
        "Continue reading\n1 no -> 5 yes"
      )
    ),
    rating = factor(as.integer(rating), levels = 5:1)
  )

part2_clean <- part2 %>%
  filter(!is.na(preferred_translation)) %>%
  mutate(
    preferred_translation = factor(preferred_translation, levels = c("HT", "MT")),
    difficulty = factor(
      difficulty,
      levels = c("similar_quality", "better", "significantly_better")
    ),
    source_lang = factor(source_lang, levels = language_levels)
  )

span_clean <- part2_span %>%
  filter(!is.na(version), !is.na(label)) %>%
  mutate(
    version = factor(version, levels = c("HT", "MT")),
    label = factor(label, levels = c("good", "poor")),
    span_chars = pmax(0, as.numeric(end) - as.numeric(start)),
    span_words = sapply(strsplit(trimws(text), "\\s+"), function(words) {
      if (length(words) == 1 && words[1] == "") 0 else length(words)
    })
  )

comparison_choice_long <- bind_rows(
  part1 %>%
    transmute(
      source_lang,
      question = "Better overall",
      choice = comparison_q1_decipher
    ),
  part1 %>%
    transmute(
      source_lang,
      question = "More varied / expressive word choice",
      choice = comparison_q2_decipher
    )
) %>%
  filter(!is.na(choice)) %>%
  mutate(
    question = factor(
      question,
      levels = c("Better overall", "More varied / expressive word choice")
    ),
    choice = factor(choice, levels = c("HT", "MT", "NO DIFF")),
    source_lang = factor(source_lang, levels = language_levels)
  )

comparison_continue <- part1 %>%
  filter(!is.na(comparison_q3), !is.na(comparison_q3_decipher)) %>%
  mutate(
    raw_choice = as.integer(comparison_q3),
    strength = case_when(
      raw_choice %in% c(1, 4) ~ "clear",
      raw_choice %in% c(2, 3) ~ "slight",
      TRUE ~ NA_character_
    ),
    choice = factor(comparison_q3_decipher, levels = c("HT", "MT")),
    fill_key = factor(
      paste(choice, strength, sep = "_"),
      levels = names(continue_strength_colors)
    ),
    source_lang = factor(source_lang, levels = language_levels)
  ) %>%
  filter(!is.na(strength), !is.na(fill_key))

comparison_ai_likely <- part1 %>%
  filter(!is.na(comparison_q5_decipher), !is.na(comparison_q6)) %>%
  mutate(
    ai_identified = factor(comparison_q5_decipher, levels = c("HT", "MT")),
    confidence = factor(as.integer(comparison_q6), levels = 1:5),
    fill_key = factor(
      paste(ai_identified, confidence, sep = "_"),
      levels = names(ai_likely_colors)
    ),
    source_lang = factor(source_lang, levels = language_levels)
  )


########### FIGURE 1: PART 1 LIKERT RATINGS ####################

likert_data <- single_long %>%
  count(question, version, rating, name = "n") %>%
  group_by(question, version) %>%
  mutate(
    total = sum(n),
    pct = 100 * n / total,
    label = percent_label(pct)
  ) %>%
  ungroup()

plot_likert <- ggplot(likert_data, aes(x = version, y = pct, fill = rating)) +
  geom_col(width = 0.72, color = "white", linewidth = 0.2) +
  geom_text(
    aes(label = label),
    position = position_stack(vjust = 0.5),
    size = 2.7,
    color = "black"
  ) +
  facet_wrap(~ question, nrow = 1) +
  scale_x_discrete(labels = c("HT" = "Human\ntranslation", "MT" = "Machine\ntranslation")) +
  scale_fill_manual(
    values = rating_colors,
    breaks = c("5", "4", "3", "2", "1"),
    labels = c("5 best", "4", "3 neutral", "2", "1 worst"),
    drop = FALSE
  ) +
  scale_y_continuous(labels = function(x) paste0(x, "%"), expand = expansion(mult = c(0, 0.04))) +
  labs(
    title = "Single-reading ratings by translation type",
    subtitle = "Worst ratings are at the bottom of each bar; best ratings are at the top.",
    x = "Part 1 questions 1-4, shown for both HT and MT",
    y = "Ordinal rating responses (1-5)",
    fill = "Rating"
  ) +
  theme_litmt() +
  theme(
    axis.text.x = element_text(size = 10, lineheight = 0.95),
    axis.title.x = element_text(margin = margin(t = 8)),
    strip.text = element_text(size = 9.5, lineheight = 1.05),
    legend.position = "bottom"
  )

save_plot(plot_likert, "part1_likert_ratings", width = 12, height = 5.4)


########### FIGURE 2: PART 1 COMPARISON CHOICES ####################

comparison_choice_summary <- comparison_choice_long %>%
  count(question, choice, name = "n") %>%
  group_by(question) %>%
  mutate(
    total = sum(n),
    pct = 100 * n / total,
    label = ifelse(pct >= 4, paste0(n, " (", percent_label(pct), ")"), "")
  ) %>%
  ungroup()

plot_comparison_choices <- ggplot(
  comparison_choice_summary,
  aes(x = question, y = pct, fill = choice)
) +
  geom_col(width = 0.68, color = "white", linewidth = 0.25) +
  geom_text(
    aes(label = label),
    position = position_stack(vjust = 0.5),
    size = 3.2,
    color = "black"
  ) +
  scale_fill_manual(values = translation_colors, drop = FALSE) +
  scale_y_continuous(labels = function(x) paste0(x, "%"), expand = expansion(mult = c(0, 0.04))) +
  coord_cartesian(ylim = c(0, 100), clip = "off") +
  labs(
    title = "Whole-excerpt comparison choices",
    subtitle = "Comparison questions asking which version was better overall or more varied / expressive.",
    x = NULL,
    y = "Responses",
    fill = "Chosen version"
  ) +
  theme_litmt() +
  theme(
    axis.text.x = element_text(size = 10, lineheight = 1.05),
    legend.position = "bottom"
  )

save_plot(plot_comparison_choices, "part1_comparison_choices", width = 8, height = 5)


comparison_choice_language <- comparison_choice_long %>%
  count(source_lang, question, choice, name = "n") %>%
  group_by(source_lang, question) %>%
  mutate(
    total = sum(n),
    pct = 100 * n / total,
    label = ifelse(pct >= 13, paste0(n, "\n", percent_label(pct)), as.character(n))
  ) %>%
  ungroup()

plot_comparison_choices_language <- ggplot(
  comparison_choice_language,
  aes(x = source_lang, y = pct, fill = choice)
) +
  geom_col(width = 0.68, color = "white", linewidth = 0.25) +
  geom_text(
    aes(label = label),
    position = position_stack(vjust = 0.5),
    size = 2.9,
    color = "black"
  ) +
  facet_wrap(~ question, nrow = 1) +
  scale_fill_manual(values = translation_colors, drop = FALSE) +
  scale_y_continuous(labels = function(x) paste0(x, "%"), expand = expansion(mult = c(0, 0.04))) +
  coord_cartesian(ylim = c(0, 100), clip = "off") +
  labs(
    title = paste("Whole-excerpt comparison choices by", language_group_label_lower),
    subtitle = paste(
      paste(language_levels, collapse = ", "),
      "groups show different HT/MT/no-difference patterns."
    ),
    x = language_group_label,
    y = "Responses",
    fill = "Chosen version"
  ) +
  theme_litmt() +
  theme(
    legend.position = "bottom",
    strip.text = element_text(size = 9.5, lineheight = 1.05)
  )

save_plot(plot_comparison_choices_language, "part1_comparison_choices_by_language", width = 10, height = 5)


########### FIGURE 3: PART 1 CONTINUE-READING PREFERENCE ####################

continue_summary <- comparison_continue %>%
  count(fill_key, name = "n") %>%
  mutate(
    total = sum(n),
    pct = 100 * n / total,
    label = ifelse(pct >= 4, paste0(n, " (", percent_label(pct), ")"), "")
  )

plot_continue_strength <- ggplot(continue_summary, aes(x = "Continue reading", y = pct, fill = fill_key)) +
  geom_col(width = 0.58, color = "white", linewidth = 0.25) +
  geom_text(
    aes(label = label),
    position = position_stack(vjust = 0.5),
    size = 3.2,
    color = "black"
  ) +
  scale_fill_manual(
    values = continue_strength_colors,
    labels = continue_strength_labels,
    drop = FALSE
  ) +
  scale_y_continuous(labels = function(x) paste0(x, "%"), expand = expansion(mult = c(0, 0.04))) +
  coord_cartesian(ylim = c(0, 100), clip = "off") +
  labs(
    title = "Preferred version for continuing the excerpt",
    subtitle = "Raw four-way answers are mapped back to HT or MT and retain slight vs clear preference.",
    x = NULL,
    y = "Responses",
    fill = "Preference"
  ) +
  theme_litmt() +
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    legend.position = "bottom"
  )

save_plot(plot_continue_strength, "part1_continue_preference_strength", width = 7, height = 5)


continue_language <- comparison_continue %>%
  count(source_lang, fill_key, name = "n") %>%
  group_by(source_lang) %>%
  mutate(
    total = sum(n),
    pct = 100 * n / total,
    label = ifelse(pct >= 13, paste0(n, "\n", percent_label(pct)), "")
  ) %>%
  ungroup()

plot_continue_language <- ggplot(continue_language, aes(x = source_lang, y = pct, fill = fill_key)) +
  geom_col(width = 0.68, color = "white", linewidth = 0.25) +
  geom_text(
    aes(label = label),
    position = position_stack(vjust = 0.5),
    size = 2.9,
    color = "black"
  ) +
  scale_fill_manual(
    values = continue_strength_colors,
    labels = continue_strength_labels,
    drop = FALSE
  ) +
  scale_y_continuous(labels = function(x) paste0(x, "%"), expand = expansion(mult = c(0, 0.04))) +
  coord_cartesian(ylim = c(0, 100), clip = "off") +
  labs(
    title = paste("Continue-reading preference by", language_group_label_lower),
    subtitle = "Preference strength is retained from the raw four-way comparison answer.",
    x = language_group_label,
    y = "Responses",
    fill = "Preference"
  ) +
  theme_litmt() +
  theme(legend.position = "bottom")

save_plot(plot_continue_language, "part1_continue_preference_by_language", width = 8, height = 5)


########### FIGURE 4: PART 1 AI-LIKELY CHOICE ####################

ai_likely_summary <- comparison_ai_likely %>%
  count(ai_identified, confidence, fill_key, name = "n") %>%
  group_by(ai_identified) %>%
  mutate(
    total = sum(n),
    pct = 100 * n / total,
    label = ifelse(pct >= 7, paste0(n, "\n", percent_label(pct)), "")
  ) %>%
  ungroup()

plot_ai_likely <- ggplot(ai_likely_summary, aes(x = ai_identified, y = pct, fill = fill_key)) +
  geom_col(width = 0.68, color = "white", linewidth = 0.25) +
  geom_text(
    aes(label = label),
    position = position_stack(vjust = 0.5),
    size = 3.0,
    color = "black"
  ) +
  scale_x_discrete(labels = c("HT" = "HT identified\nas AI", "MT" = "MT identified\nas AI")) +
  scale_fill_manual(
    values = ai_likely_colors,
    breaks = names(ai_likely_colors),
    labels = ai_likely_labels,
    drop = FALSE
  ) +
  scale_y_continuous(labels = function(x) paste0(x, "%"), expand = expansion(mult = c(0, 0.04))) +
  coord_cartesian(ylim = c(0, 100), clip = "off") +
  labs(
    title = "Which whole-excerpt version looked AI-translated?",
    subtitle = "Bars split the chosen version by confidence; darker shades indicate higher confidence.",
    x = NULL,
    y = "Responses within identified version",
    fill = "Version + confidence"
  ) +
  theme_litmt() +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 8.5)
  ) +
  guides(fill = guide_legend(nrow = 5, byrow = TRUE))

save_plot(plot_ai_likely, "part1_ai_likely_confidence", width = 8.5, height = 6.2)


ai_likely_language <- comparison_ai_likely %>%
  count(source_lang, ai_identified, confidence, fill_key, name = "n") %>%
  group_by(source_lang) %>%
  mutate(
    total = sum(n),
    pct = 100 * n / total,
    signed_pct = ifelse(ai_identified == "HT", -pct, pct)
  ) %>%
  ungroup()

plot_ai_likely_language <- ggplot(
  ai_likely_language,
  aes(x = source_lang, y = signed_pct, fill = fill_key)
) +
  geom_col(width = 0.68, color = "white", linewidth = 0.25) +
  geom_hline(yintercept = 0, color = "gray30", linewidth = 0.4) +
  scale_fill_manual(
    values = ai_likely_colors,
    breaks = names(ai_likely_colors),
    labels = ai_likely_labels,
    drop = FALSE
  ) +
  scale_y_continuous(
    labels = function(x) paste0(abs(x), "%"),
    breaks = seq(-100, 100, 25),
    limits = c(-100, 100)
  ) +
  labs(
    title = paste("AI-likely whole-excerpt choices by", language_group_label_lower),
    subtitle = "Left bars identify HT as AI; right bars identify MT as AI. Darker shades indicate higher confidence.",
    x = language_group_label,
    y = "AI-likely responses",
    fill = "Version + confidence"
  ) +
  theme_litmt() +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 8.5)
  ) +
  guides(fill = guide_legend(nrow = 5, byrow = TRUE))

save_plot(plot_ai_likely_language, "part1_ai_likely_by_language", width = 9, height = 6.2)


########### FIGURE 2: PART 2 PREFERENCE BY SOURCE LANGUAGE ####################

pref_lang_strength <- part2_clean %>%
  mutate(
    source_lang = factor(source_lang, levels = language_levels),
    preferred_translation = factor(preferred_translation, levels = c("HT", "MT")),
    difficulty = factor(
      difficulty,
      levels = c("similar_quality", "better", "significantly_better")
    ),
    fill_key = factor(
      paste(preferred_translation, difficulty, sep = "_"),
      levels = c(
        "HT_similar_quality",
        "HT_better",
        "HT_significantly_better",
        "MT_similar_quality",
        "MT_better",
        "MT_significantly_better"
      )
    )
  ) %>%
  count(source_lang, preferred_translation, difficulty, fill_key, name = "n") %>%
  group_by(source_lang) %>%
  mutate(
    total = sum(n),
    pct = 100 * n / total
  ) %>%
  group_by(source_lang, preferred_translation) %>%
  arrange(source_lang, preferred_translation, difficulty, .by_group = TRUE) %>%
  mutate(
    bar_total_n = sum(n),
    bar_total_pct = sum(pct),
    ymin = cumsum(pct) - pct,
    ymax = cumsum(pct),
    label_y = ymin + pct / 2,
    segment_label = ifelse(pct >= 6, paste0(n, "\n", percent_label(pct)), ""),
    label_color = ifelse(difficulty == "significantly_better", "white", "black"),
    x_center = as.numeric(source_lang) + ifelse(preferred_translation == "HT", -0.19, 0.19),
    xmin = x_center - 0.17,
    xmax = x_center + 0.17
  ) %>%
  ungroup()

pref_lang_totals <- pref_lang_strength %>%
  group_by(source_lang, preferred_translation, x_center, bar_total_n, bar_total_pct) %>%
  summarise(label_y = max(ymax), .groups = "drop") %>%
  mutate(
    label_y = label_y + 2.2,
    label = paste0(bar_total_n, " (", percent_label(bar_total_pct), ")")
  )

plot_pref_lang <- ggplot(pref_lang_strength) +
  geom_rect(
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = fill_key),
    color = "white",
    linewidth = 0.25
  ) +
  geom_text(
    aes(x = x_center, y = label_y, label = segment_label, color = label_color),
    size = 2.45,
    lineheight = 0.86,
    fontface = "bold"
  ) +
  geom_text(
    data = pref_lang_totals,
    aes(x = x_center, y = label_y, label = label),
    vjust = 0,
    size = 3.2
  ) +
  scale_fill_manual(
    values = preference_difficulty_colors,
    labels = preference_difficulty_labels,
    drop = FALSE
  ) +
  scale_color_identity() +
  scale_x_continuous(
    breaks = seq_along(levels(pref_lang_strength$source_lang)),
    labels = levels(pref_lang_strength$source_lang)
  ) +
  scale_y_continuous(labels = function(x) paste0(x, "%"), expand = expansion(mult = c(0, 0.08))) +
  coord_cartesian(ylim = c(0, 100), clip = "off") +
  labs(
    title = paste("Chunk-level preference by", language_group_label_lower),
    subtitle = "Bars keep the HT/MT preference share and are subdivided by how strong the preference was.",
    x = language_group_label,
    y = "Preferred in chunk comparisons",
    fill = "Preference strength"
  ) +
  theme_litmt() +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 8.5)
  ) +
  guides(fill = guide_legend(nrow = 2, byrow = TRUE))

save_plot(plot_pref_lang, "part2_preference_by_language", width = 8, height = 5)


########### FIGURE 2B: PART 2 PREFERENCE BY DIFFICULTY ####################

pref_difficulty <- part2_clean %>%
  mutate(
    fill_key = factor(
      paste(preferred_translation, difficulty, sep = "_"),
      levels = c(
        "HT_significantly_better",
        "HT_better",
        "HT_similar_quality",
        "MT_significantly_better",
        "MT_better",
        "MT_similar_quality"
      )
    )
  ) %>%
  count(fill_key, preferred_translation, difficulty, name = "n") %>%
  mutate(
    total = sum(n),
    pct = 100 * n / total,
    signed_pct = ifelse(preferred_translation == "MT", -pct, pct),
    label = percent_label(pct),
    label_color = ifelse(
      preferred_translation == "HT" & difficulty == "significantly_better",
      "white",
      "black"
    ),
    strength_rank = as.integer(factor(
      difficulty,
      levels = c("similar_quality", "better", "significantly_better")
    ))
  ) %>%
  arrange(preferred_translation, strength_rank) %>%
  group_by(preferred_translation) %>%
  mutate(
    cumulative_pct = cumsum(pct),
    label_y = case_when(
      preferred_translation == "HT" ~ cumulative_pct - pct / 2,
      preferred_translation == "MT" ~ -(cumulative_pct - pct / 2),
      TRUE ~ NA_real_
    )
  ) %>%
  ungroup()

plot_pref_difficulty <- ggplot(pref_difficulty, aes(x = "Chunk choices", y = signed_pct, fill = fill_key)) +
  geom_col(width = 0.58, color = "white", linewidth = 0.25) +
  geom_text(
    aes(y = label_y, label = label, color = label_color),
    size = 3.0,
    lineheight = 0.9,
    fontface = "bold"
  ) +
  geom_hline(yintercept = 0, color = "gray30", linewidth = 0.4) +
  scale_fill_manual(
    values = preference_difficulty_colors,
    labels = preference_difficulty_labels,
    drop = FALSE
  ) +
  scale_color_identity() +
  scale_y_continuous(
    labels = function(x) paste0(abs(x), "%"),
    breaks = seq(-100, 100, 25),
    limits = c(-100, 100)
  ) +
  labs(
    title = "Chunk-level preference strength",
    subtitle = "Left bars are MT preferences; right bars are HT preferences. Darker shades indicate clearer choices.",
    x = NULL,
    y = "Preference share",
    fill = "Preference strength"
  ) +
  theme_litmt() +
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    legend.position = "bottom"
  )

save_plot(plot_pref_difficulty, "part2_preference_by_difficulty", width = 8, height = 5)


pref_difficulty_language <- part2_clean %>%
  mutate(
    fill_key = factor(
      paste(preferred_translation, difficulty, sep = "_"),
      levels = c(
        "HT_significantly_better",
        "HT_better",
        "HT_similar_quality",
        "MT_significantly_better",
        "MT_better",
        "MT_similar_quality"
      )
    )
  ) %>%
  count(source_lang, fill_key, preferred_translation, difficulty, name = "n") %>%
  group_by(source_lang) %>%
  mutate(
    total = sum(n),
    pct = 100 * n / total,
    signed_pct = ifelse(preferred_translation == "MT", -pct, pct),
    label = percent_label(pct),
    label_color = ifelse(
      preferred_translation == "HT" & difficulty == "significantly_better",
      "white",
      "black"
    ),
    strength_rank = as.integer(factor(
      difficulty,
      levels = c("similar_quality", "better", "significantly_better")
    ))
  ) %>%
  arrange(source_lang, preferred_translation, strength_rank) %>%
  group_by(source_lang, preferred_translation) %>%
  mutate(
    cumulative_pct = cumsum(pct),
    label_y = case_when(
      preferred_translation == "HT" ~ cumulative_pct - pct / 2,
      preferred_translation == "MT" ~ -(cumulative_pct - pct / 2),
      TRUE ~ NA_real_
    )
  ) %>%
  ungroup()

plot_pref_difficulty_language <- ggplot(
  pref_difficulty_language,
  aes(x = source_lang, y = signed_pct, fill = fill_key)
) +
  geom_col(width = 0.68, color = "white", linewidth = 0.25) +
  geom_text(
    aes(y = label_y, label = label, color = label_color),
    size = 2.6,
    lineheight = 0.9,
    fontface = "bold"
  ) +
  geom_hline(yintercept = 0, color = "gray30", linewidth = 0.4) +
  scale_fill_manual(
    values = preference_difficulty_colors,
    labels = preference_difficulty_labels,
    drop = FALSE
  ) +
  scale_color_identity() +
  scale_y_continuous(
    labels = function(x) paste0(abs(x), "%"),
    breaks = seq(-100, 100, 25),
    limits = c(-100, 100)
  ) +
  labs(
    title = paste("Chunk-level preference strength by", language_group_label_lower),
    subtitle = "Left bars are MT preferences; right bars are HT preferences. Darker shades indicate clearer choices.",
    x = language_group_label,
    y = "Preference share",
    fill = "Preference strength"
  ) +
  theme_litmt() +
  theme(legend.position = "bottom")

save_plot(plot_pref_difficulty_language, "part2_preference_difficulty_by_language", width = 9, height = 5.5)


########### FIGURE 3: BOOK-LEVEL DIVERGING PREFERENCE BY ANNOTATOR ####################

book_pref <- part2_clean %>%
  mutate(
    book_label = gsub("^(french|japanese|polish)_eval_(FIXED_)?", "", book_id),
    book_label = gsub("_", " ", book_label),
    reader_short = user_id,
    fill_key = factor(
      paste(preferred_translation, difficulty, sep = "_"),
      levels = names(preference_difficulty_colors)
    )
  ) %>%
  count(source_lang, book_label, reader_short, preferred_translation, difficulty, fill_key, name = "n") %>%
  group_by(source_lang, book_label, reader_short) %>%
  mutate(
    total = sum(n),
    pct = 100 * n / total,
    signed_pct = ifelse(preferred_translation == "MT", -pct, pct)
  ) %>%
  ungroup()

book_order <- book_pref %>%
  group_by(book_label) %>%
  summarise(ht_margin = sum(signed_pct), .groups = "drop") %>%
  arrange(ht_margin) %>%
  pull(book_label)

reader_order <- book_pref %>%
  group_by(book_label, reader_short) %>%
  summarise(ht_margin = sum(signed_pct), .groups = "drop") %>%
  mutate(
    book_label = factor(book_label, levels = book_order),
    plot_label = paste0(book_label, "  |  ", reader_short)
  ) %>%
  arrange(book_label, ht_margin) %>%
  pull(plot_label)

book_pref <- book_pref %>%
  mutate(
    book_label = factor(book_label, levels = book_order),
    plot_label = paste0(book_label, "  |  ", reader_short)
  )

plot_book_pref <- ggplot(
  book_pref,
  aes(x = factor(plot_label, levels = reader_order), y = signed_pct, fill = fill_key)
) +
  geom_col(width = 0.72, color = "white", linewidth = 0.15, position = position_stack(reverse = TRUE)) +
  coord_flip(clip = "off") +
  geom_hline(yintercept = 0, color = "gray30", linewidth = 0.4) +
  scale_fill_manual(
    values = preference_difficulty_colors,
    labels = preference_difficulty_labels,
    drop = FALSE
  ) +
  scale_y_continuous(
    labels = function(x) paste0(abs(x), "%"),
    breaks = seq(-100, 100, 25),
    limits = c(-100, 100)
  ) +
  labs(
    title = "Book-level preference by annotator",
    subtitle = paste0(
      "Each row is one annotator for one book.\n",
      "Left bars indicate MT preference; right bars indicate HT preference. ",
      "Darker shades indicate clearer preferences."
    ),
    x = "Book | annotator",
    y = "Preference share",
    fill = "Preference strength"
  ) +
  theme_litmt(base_size = 10) +
  theme(
    plot.subtitle = element_text(size = 9.5, lineheight = 1.08),
    plot.margin = margin(10, 18, 10, 18)
  )

save_plot(plot_book_pref, "book_preference_diverging", width = 11, height = 10)


########### FIGURE 4: BOOK-LEVEL AI IDENTIFICATION ####################

book_ai_id <- part1 %>%
  filter(!is.na(comparison_q5_decipher), !is.na(comparison_q6)) %>%
  mutate(
    book_label = gsub("^(french|japanese|polish)_eval_(FIXED_)?", "", book_id),
    book_label = gsub("_", " ", book_label),
    ai_identified = factor(comparison_q5_decipher, levels = c("HT", "MT")),
    confidence = factor(as.integer(comparison_q6), levels = 1:5),
    fill_key = factor(
      paste(ai_identified, confidence, sep = "_"),
      levels = ai_identification_stack_order
    )
  ) %>%
  count(source_lang, book_label, ai_identified, confidence, fill_key, name = "n") %>%
  group_by(source_lang, book_label) %>%
  mutate(
    total = sum(n),
    pct = 100 * n / total,
    signed_pct = ifelse(ai_identified == "HT", -pct, pct)
  ) %>%
  ungroup()

ai_book_order <- book_ai_id %>%
  group_by(source_lang, book_label) %>%
  summarise(mt_ai_margin = sum(signed_pct), .groups = "drop") %>%
  arrange(source_lang, mt_ai_margin, book_label) %>%
  pull(book_label)

ai_language_boundaries <- book_ai_id %>%
  distinct(source_lang, book_label) %>%
  mutate(book_label = factor(book_label, levels = ai_book_order)) %>%
  arrange(book_label) %>%
  mutate(
    row_index = row_number(),
    next_source_lang = lead(source_lang)
  ) %>%
  filter(!is.na(next_source_lang), source_lang != next_source_lang) %>%
  transmute(yintercept = row_index + 0.5)

ai_language_labels <- book_ai_id %>%
  distinct(source_lang, book_label) %>%
  mutate(book_label = factor(book_label, levels = ai_book_order)) %>%
  arrange(book_label) %>%
  group_by(source_lang) %>%
  summarise(row_midpoint = mean(as.numeric(book_label)), .groups = "drop")

book_ai_id <- book_ai_id %>%
  mutate(
    book_label = factor(book_label, levels = ai_book_order)
  )

plot_book_ai_id <- ggplot(
  book_ai_id,
  aes(x = book_label, y = signed_pct, fill = fill_key)
) +
  geom_col(width = 0.72, color = "white", linewidth = 0.15) +
  geom_vline(
    data = ai_language_boundaries,
    aes(xintercept = yintercept),
    inherit.aes = FALSE,
    color = "gray45",
    linewidth = 0.35,
    linetype = "dotted"
  ) +
  geom_text(
    data = ai_language_labels,
    aes(x = row_midpoint, y = -113, label = source_lang),
    inherit.aes = FALSE,
    fontface = "bold",
    hjust = 0,
    size = 3.5
  ) +
  coord_flip(ylim = c(-115, 100), clip = "off") +
  geom_hline(yintercept = 0, color = "gray30", linewidth = 0.4) +
  scale_fill_manual(
    values = ai_identification_colors,
    breaks = names(ai_identification_colors),
    labels = ai_identification_labels,
    drop = FALSE
  ) +
  scale_y_continuous(
    labels = function(x) paste0(abs(x), "%"),
    breaks = seq(-100, 100, 25),
    limits = c(-115, 100)
  ) +
  labs(
    title = "Book-level AI identification accuracy",
    subtitle = paste0(
      "Each row is one book, aggregated across participants.\n",
      "Left bars incorrectly identify HT as AI; right bars correctly identify MT as AI. ",
      "Darker shades indicate higher confidence."
    ),
    x = "Book",
    y = "AI identification responses",
    fill = "Incorrect (orange) / Correct (green)"
  ) +
  theme_litmt(base_size = 10) +
  theme(
    plot.subtitle = element_text(size = 9.5, lineheight = 1.08),
    legend.position = "right",
    legend.text = element_text(size = 8.5),
    plot.margin = margin(10, 18, 10, 18)
  ) +
  guides(fill = guide_legend(ncol = 1))

save_plot(plot_book_ai_id, "book_ai_identification_diverging", width = 11, height = 7)


########### FIGURE 5: SPAN HIGHLIGHTS ####################

span_counts <- span_clean %>%
  count(version, label, name = "n") %>%
  group_by(version) %>%
  mutate(
    total = sum(n),
    pct = 100 * n / total,
    label_text = paste0(n, " (", percent_label(pct), ")")
  ) %>%
  ungroup()

plot_spans <- ggplot(span_counts, aes(x = version, y = pct, fill = label)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.68) +
  geom_text(
    aes(label = label_text),
    position = position_dodge(width = 0.75),
    vjust = -0.35,
    size = 3.3
  ) +
  scale_fill_manual(values = highlight_colors, drop = FALSE) +
  scale_y_continuous(labels = function(x) paste0(x, "%"), expand = expansion(mult = c(0, 0.08))) +
  coord_cartesian(ylim = c(0, 100), clip = "off") +
  labs(
    title = "Highlighted spans by translation type",
    subtitle = "Participants marked spans they liked or disliked in HT and MT chunks",
    x = "Highlighted version",
    y = "Highlighted spans",
    fill = "Span label"
  ) +
  theme_litmt()

save_plot(plot_spans, "span_highlights_by_version", width = 7.5, height = 5)


########### FIGURE 5B: SPAN HIGHLIGHT LENGTHS ####################

span_length_clean <- span_clean %>%
  filter(!is.na(span_words), span_words > 0, span_words <= 80)

plot_span_lengths <- ggplot(
  span_length_clean,
  aes(x = label, y = span_words, fill = version)
) +
  geom_boxplot(width = 0.62, outlier.alpha = 0.18, outlier.size = 0.8) +
  scale_fill_manual(values = translation_colors, drop = FALSE) +
  scale_y_continuous(expand = expansion(mult = c(0.02, 0.08))) +
  labs(
    title = "Highlighted span length by label and translation type",
    subtitle = "Very long highlights above 80 words are excluded to keep the distribution readable.",
    x = "Span label",
    y = "Highlighted span length (words)",
    fill = "Version"
  ) +
  theme_litmt()

save_plot(plot_span_lengths, "span_highlight_lengths", width = 8, height = 5)


span_length_language <- span_clean %>%
  filter(!is.na(span_words), span_words > 0) %>%
  group_by(source_lang, version, label) %>%
  summarise(
    median_words = median(span_words, na.rm = TRUE),
    mean_words = mean(span_words, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  mutate(
    source_lang = factor(source_lang, levels = language_levels),
    label_text = paste0("n=", n)
  )

plot_span_length_language <- ggplot(
  span_length_language,
  aes(x = source_lang, y = median_words, fill = version)
) +
  geom_col(position = position_dodge(width = 0.75), width = 0.68) +
  geom_text(
    aes(label = label_text),
    position = position_dodge(width = 0.75),
    vjust = -0.35,
    size = 2.8
  ) +
  facet_wrap(~ label, nrow = 1) +
  scale_fill_manual(values = translation_colors, drop = FALSE) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(
    title = paste("Median highlighted span length by", language_group_label_lower),
    subtitle = paste(
      "Highlight patterns can differ by",
      language_group_label_lower,
      "translation type, label, and length."
    ),
    x = language_group_label,
    y = "Median highlighted span length (words)",
    fill = "Version"
  ) +
  theme_litmt() +
  theme(legend.position = "bottom")

save_plot(plot_span_length_language, "span_highlight_median_length_by_language", width = 9, height = 5)


span_density <- span_clean %>%
  count(source_lang, version, label, name = "n") %>%
  group_by(source_lang, version) %>%
  mutate(
    total = sum(n),
    pct = 100 * n / total,
    label_text = ifelse(pct >= 8, paste0(n, "\n", percent_label(pct)), "")
  ) %>%
  ungroup() %>%
  mutate(source_lang = factor(source_lang, levels = language_levels))

plot_span_density <- ggplot(span_density, aes(x = version, y = pct, fill = label)) +
  geom_col(width = 0.68, color = "white", linewidth = 0.25) +
  geom_text(
    aes(label = label_text),
    position = position_stack(vjust = 0.5),
    size = 2.9,
    color = "black"
  ) +
  facet_wrap(~ source_lang, nrow = 1) +
  scale_fill_manual(values = highlight_colors, drop = FALSE) +
  scale_y_continuous(labels = function(x) paste0(x, "%"), expand = expansion(mult = c(0, 0.04))) +
  coord_cartesian(ylim = c(0, 100), clip = "off") +
  labs(
    title = paste("Good vs poor highlights by", language_group_label_lower),
    subtitle = paste(
      "Within each",
      language_group_label_lower,
      "and translation type, bars show the split of liked vs disliked spans."
    ),
    x = "Highlighted version",
    y = "Highlighted spans",
    fill = "Span label"
  ) +
  theme_litmt() +
  theme(legend.position = "bottom")

save_plot(plot_span_density, "span_highlights_by_language", width = 9, height = 5)


########### FIGURE 5: ORIGIN GUESS CONFUSION ####################

origin_guess <- bind_rows(
  part1 %>% transmute(actual = first_version, guessed = first_q7_decipher),
  part1 %>% transmute(actual = second_version, guessed = second_q7_decipher)
) %>%
  filter(!is.na(actual), !is.na(guessed)) %>%
  mutate(
    actual = factor(actual, levels = c("HT", "MT")),
    guessed = factor(guessed, levels = c("HT", "MT"))
  ) %>%
  count(actual, guessed, name = "n") %>%
  group_by(actual) %>%
  mutate(pct = 100 * n / sum(n), label = paste0(n, "\n", percent_label(pct))) %>%
  ungroup()

plot_origin <- ggplot(origin_guess, aes(x = guessed, y = actual, fill = pct)) +
  geom_tile(color = "white", linewidth = 0.8) +
  geom_text(aes(label = label), size = 4) +
  scale_fill_gradient(low = "#F7F7F7", high = "#2166AC", limits = c(0, 100)) +
  labs(
    title = "Readers struggle to identify translation origin",
    subtitle = "Actual version vs participant guess in single-reading questions",
    x = "Guessed version",
    y = "Actual version",
    fill = "Row %"
  ) +
  theme_litmt()

save_plot(plot_origin, "origin_guess_confusion", width = 6, height = 5)


########### FIGURE 6: ORIGIN GUESS ACCURACY BY READING ####################

origin_accuracy <- bind_rows(
  part1 %>%
    transmute(
      reading = "First reading",
      actual = first_version,
      guessed = first_q7_decipher,
      confidence = first_q8
    ),
  part1 %>%
    transmute(
      reading = "Second reading",
      actual = second_version,
      guessed = second_q7_decipher,
      confidence = second_q8
    )
) %>%
  filter(!is.na(actual), !is.na(guessed), !is.na(confidence)) %>%
  mutate(
    reading = factor(reading, levels = c("First reading", "Second reading")),
    actual = factor(actual, levels = c("HT", "MT")),
    guess_status = case_when(
      guessed == actual ~ "Correct",
      TRUE ~ "Incorrect"
    ),
    guess_status = factor(guess_status, levels = c("Incorrect", "Correct")),
    confidence = factor(as.integer(confidence), levels = 1:5),
    fill_key = factor(
      paste(guess_status, confidence, sep = "_"),
      levels = origin_accuracy_stack_order
    )
  ) %>%
  count(reading, actual, guess_status, confidence, fill_key, name = "n") %>%
  group_by(reading, actual) %>%
  mutate(
    total = sum(n),
    pct = 100 * n / total,
    label = ifelse(n > 0, paste0(n, " (", percent_label(pct), ")"), "")
  ) %>%
  ungroup()

plot_origin_accuracy <- ggplot(
  origin_accuracy,
  aes(x = actual, y = pct, fill = fill_key)
) +
  geom_col(width = 0.68, color = "white", linewidth = 0.25) +
  geom_text(
    aes(label = label),
    position = position_stack(vjust = 0.5),
    size = 3.2,
    color = "black"
  ) +
  facet_wrap(~ reading, nrow = 1) +
  scale_x_discrete(labels = c(
    "HT" = "Human translation",
    "MT" = "Machine translation"
  )) +
  scale_fill_manual(
    values = origin_accuracy_colors,
    breaks = names(origin_accuracy_colors),
    labels = origin_accuracy_labels,
    drop = FALSE
  ) +
  scale_y_continuous(labels = function(x) paste0(x, "%"), expand = expansion(mult = c(0, 0.04))) +
  coord_cartesian(ylim = c(0, 100), clip = "off") +
  labs(
    title = "Translation-origin guess accuracy and confidence by reading",
    subtitle = paste(
      "Blank Q7/Q8 rows are excluded.",
      "Darker shades indicate higher confidence."
    ),
    x = "Actual version shown",
    y = "Single-reading origin guesses",
    fill = "Correct (green) / Incorrect (orange)"
  ) +
  theme_litmt() +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 8.5),
    strip.text = element_text(size = 10)
  ) +
  guides(fill = guide_legend(nrow = 2, byrow = TRUE))

save_plot(plot_origin_accuracy, "origin_guess_accuracy_by_reading", width = 9.5, height = 5.4)


########### FIGURE 7: PART 2 CHUNK CHOICE HEATMAP ####################

chunk_choices <- part2_clean %>%
  mutate(
    chunk_id = as.integer(chunk_id),
    book_label = gsub("^(french|japanese|polish)_eval_(FIXED_)?", "", book_id),
    book_label = gsub("_", " ", book_label),
    reader_short = user_id,
    row_label = paste0(book_label, "  |  ", reader_short),
    fill_key = factor(
      paste(preferred_translation, difficulty, sep = "_"),
      levels = names(choice_heatmap_colors)
    )
  ) %>%
  group_by(book_label, reader_short) %>%
  mutate(ht_share = mean(preferred_translation == "HT", na.rm = TRUE)) %>%
  ungroup()

heatmap_row_order <- chunk_choices %>%
  distinct(source_lang, book_label, reader_short, row_label, ht_share) %>%
  arrange(source_lang, book_label, desc(ht_share), reader_short) %>%
  pull(row_label)

chunk_choices <- chunk_choices %>%
  mutate(row_label = factor(row_label, levels = rev(unique(heatmap_row_order))))

max_chunk_id <- max(chunk_choices$chunk_id, na.rm = TRUE)

plot_chunk_heatmap <- ggplot(
  chunk_choices,
  aes(x = chunk_id, y = row_label, fill = fill_key)
) +
  geom_tile(color = "white", linewidth = 0.25, height = 0.86) +
  scale_fill_manual(
    values = choice_heatmap_colors,
    labels = choice_heatmap_labels,
    drop = FALSE
  ) +
  scale_x_continuous(
    breaks = seq(0, max_chunk_id, by = 5),
    expand = expansion(mult = c(0.005, 0.02))
  ) +
  labs(
    title = "Part 2 chunk-level choices by annotator",
    subtitle = "Each cell is one chunk comparison. Darker green indicates stronger HT preference; darker purple indicates stronger MT preference.",
    x = "Chunk ID",
    y = NULL,
    fill = "Preferred + strength"
  ) +
  theme_litmt(base_size = 9.5) +
  theme(
    axis.text.y = element_text(size = 7.5),
    panel.grid = element_blank(),
    legend.position = "bottom",
    plot.margin = margin(10, 14, 10, 14)
  ) +
  guides(fill = guide_legend(nrow = 2, byrow = TRUE))

save_plot(plot_chunk_heatmap, "part2_chunk_choice_heatmap", width = 12, height = 10)

cat("Saved figures to:", figure_dir, "\n")
