#!/bin/bash
#SBATCH -J agents_pipeline
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4-00:00
#SBATCH --nodelist=cs-venus-12
#SBATCH --output=/home/yferstle/litmt/LitMT/logs/%x-%j.out
#SBATCH --error=/home/yferstle/litmt/LitMT/logs/%x-%j.err

set -euo pipefail
umask 077

PROJECT_ROOT="/home/yferstle/litmt/LitMT"
LOG_DIR="${PROJECT_ROOT}/logs"
BOOK_DIR="${PROJECT_ROOT}/books/eval"
REAL_HOME="${HOME}"
JOB_HOME_ROOT="${PROJECT_ROOT}/.batch-home-${SLURM_JOB_ID:-$$}"

cleanup() {
  if [[ -d "${JOB_HOME_ROOT}" ]]; then
    rm -rf "${JOB_HOME_ROOT}"
  fi
}
trap cleanup EXIT

mkdir -p "${LOG_DIR}"

source /home/yferstle/miniconda3/etc/profile.d/conda.sh
conda activate litmt

export PATH="/home/yferstle/.nvm/versions/node/v24.14.1/bin:$PATH"
export PYTHONUNBUFFERED=1

# Retry transient provider failures instead of aborting the whole phase on the
# first Claude/Codex subprocess error.
export JOB_EXEC_RETRY_MAX="${JOB_EXEC_RETRY_MAX:-2}"
export JOB_EXEC_RETRY_BASE_SLEEP_SECONDS="${JOB_EXEC_RETRY_BASE_SLEEP_SECONDS:-10}"
export JOB_EXEC_RETRY_MAX_SLEEP_SECONDS="${JOB_EXEC_RETRY_MAX_SLEEP_SECONDS:-60}"

# This pipeline uses CLI logins, not API keys. Snapshot both tools' auth/config
# into a job-private HOME so interactive sessions cannot mutate credentials
# underneath long-running batch subprocesses.
if [[ ! -f "${REAL_HOME}/.claude/.credentials.json" ]]; then
  echo "Missing Claude credentials: ${REAL_HOME}/.claude/.credentials.json" >&2
  exit 1
fi

if [[ ! -f "${REAL_HOME}/.claude.json" ]]; then
  echo "Missing Claude config: ${REAL_HOME}/.claude.json" >&2
  exit 1
fi

if [[ ! -f "${REAL_HOME}/.codex/auth.json" ]]; then
  echo "Missing Codex credentials: ${REAL_HOME}/.codex/auth.json" >&2
  exit 1
fi

if [[ ! -f "${REAL_HOME}/.codex/config.toml" ]]; then
  echo "Missing Codex config: ${REAL_HOME}/.codex/config.toml" >&2
  exit 1
fi

rm -rf "${JOB_HOME_ROOT}"
mkdir -p "${JOB_HOME_ROOT}/.claude" "${JOB_HOME_ROOT}/.codex"

cp -a "${REAL_HOME}/.claude/.credentials.json" "${JOB_HOME_ROOT}/.claude/"
if [[ -f "${REAL_HOME}/.claude/settings.json" ]]; then
  cp -a "${REAL_HOME}/.claude/settings.json" "${JOB_HOME_ROOT}/.claude/"
fi
cp -a "${REAL_HOME}/.claude.json" "${JOB_HOME_ROOT}/.claude.json"

cp -a "${REAL_HOME}/.codex/auth.json" "${JOB_HOME_ROOT}/.codex/"
cp -a "${REAL_HOME}/.codex/config.toml" "${JOB_HOME_ROOT}/.codex/"
if [[ -f "${REAL_HOME}/.codex/version.json" ]]; then
  cp -a "${REAL_HOME}/.codex/version.json" "${JOB_HOME_ROOT}/.codex/"
fi
if [[ -f "${REAL_HOME}/.codex/installation_id" ]]; then
  cp -a "${REAL_HOME}/.codex/installation_id" "${JOB_HOME_ROOT}/.codex/"
fi

export HOME="${JOB_HOME_ROOT}"
unset CLAUDE_CODE_SIMPLE

cd "${PROJECT_ROOT}"

echo "========================================"
echo "Project root: ${PROJECT_ROOT}"
echo "Started at: $(date)"
echo "Python: $(command -v python)"
echo "Node: $(command -v node)"
echo "Claude: $(command -v claude)"
echo "Claude version: $(claude --version)"
echo "Claude auth ok"
claude auth status --json
echo "Codex: $(command -v codex)"
echo "Codex version: $(codex --version)"
echo "Codex auth: $(codex login status)"
echo "JOB_EXEC_RETRY_MAX=${JOB_EXEC_RETRY_MAX}"
echo "========================================"

mapfile -t BOOK_PATHS < <(find "${BOOK_DIR}" -type f -name "*.txt" | sort)

if [[ ${#BOOK_PATHS[@]} -eq 0 ]]; then
  echo "No .txt files found in ${BOOK_DIR}" >&2
  exit 1
fi

for BOOK_PATH in "${BOOK_PATHS[@]}"; do
  echo "========================================"
  echo "Starting: ${BOOK_PATH}"
  echo "Started at: $(date)"
  echo "========================================"

  if [[ ! -f "${BOOK_PATH}" ]]; then
    echo "File not found: ${BOOK_PATH}" >&2
    exit 1
  fi

  time srun --export=ALL python -u -m agents_pipeline.runner --book "${BOOK_PATH}"

  echo "Finished: ${BOOK_PATH}"
  echo "Finished at: $(date)"
  echo
done
