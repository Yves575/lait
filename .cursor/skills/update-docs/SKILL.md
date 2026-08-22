---
name: update-docs
description: >-
  Updates LAIT documentation in the same change when the agent makes a
  significant code, pipeline, CLI, eval, or data-layout change. Use after
  implementing features, changing pipelines, adding scripts, changing eval
  protocol, changing paths, preparing significant PRs, or when the user asks
  to update the docs.
---

# Update docs

If the change is significant, edit the owning doc **in the same change**. Match existing tone: short, table-heavy, public-vs-withheld explicit. Do not rewrite unrelated sections.

## Significant (must update)

- New module, package, or top-level directory
- Changed public CLI, flags, defaults, or entrypoint (`mt_pipeline.py`, `agents_pipeline/runner.py`, `mt_eval.py`, `eval_pipeline.py`, restore/sanitize scripts)
- New script that is a real workflow (not a one-off scratch file)
- Changed data paths, restore layout, withheld/sanitized boundary
- Changed eval protocol, reported metrics, or result-directory meaning
- Changed setup steps, required env, or how to rerun
- Architecture / pipeline stage / config behavior a reader or agent would rely on

## Not significant (skip)

- Typos, comments, formatting, dead-code cleanup
- Tiny refactors or internal helper renames with no user-facing effect
- Tests-only changes
- One-off analysis experiments that are not an entrypoint

## Which file

1. Use skill `project-docs` (topic table). Prefer editing an existing owner over adding a file.
2. Grep `docs/` and directory `README.md` files for the module/script/path name.
3. Typical owners:

| Change | Update |
| --- | --- |
| New/renamed top-level path | `docs/REPO_MAP.md`; reader routes in `docs/NAVIGATION.md` and `START_HERE.md` if it is public material |
| Setup, CLI list, env | `docs/REPRODUCIBILITY.md` and root `README.md` (keep the two in sync) |
| Withheld/public boundary, restore | `docs/DATA_ACCESS.md`; manifests under `docs/release/` only if the audit set actually changed |
| P3 stages, chunking, prompts, config | `agents_pipeline/README.md` |
| P1/P2 outputs layout | `books/MT/README.md` |
| Automatic metrics / result dirs | matching `results_*/README.md`; `docs/NAVIGATION.md` if the preprint map breaks |
| Human-eval public surface | `human_eval/README.md`, `analysis/human_eval/README.md` |
| Comment-coding process | `docs/COMMENT_CODING_SCHEMA_GUIDELINE.md` |
| Analysis script entrypoints | `analysis/README.md` |
| Publish process | `docs/RELEASE_CHECKLIST.md` |

4. If nothing owns it and it is now a real entrypoint: add a short section to the closest existing doc. Add a **new** `docs/` file only when the map would have nowhere to point.
5. If you added a docs file or a new topic, update `.cursor/skills/project-docs/SKILL.md` topic table (and `docs/REPO_MAP.md` / `docs/NAVIGATION.md` as needed).

## How to write

- Same terminology as current docs: P1/P2/P3, controlled-access, withheld, sanitized, source/HT, human evaluation, automatic metrics.
- Paths: forward slashes, repo-root relative.
- Do not copy large code into docs. Point at the CLI or directory README.
- Do not claim Hugging Face URLs or venv directory names unless you verified them (root README still has a placeholder dataset URL; agent Python env is `./venv`, not `.venv`).
- Public-release edits: do not document restored source/HT paths as if they ship on GitHub. If outputs gained text-bearing fields, follow `docs/RELEASE_CHECKLIST.md` and `scripts/sanitize_public_release_outputs.py`.
