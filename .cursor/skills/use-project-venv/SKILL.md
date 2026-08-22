---
name: use-project-venv
description: >-
  Always use this repository's Python virtual environment via
  .\venv\Scripts\Activate.ps1 (or the matching python.exe). Use whenever
  running Python, pip, pytest, notebooks, analysis scripts, or any shell
  command that needs the project interpreter.
---

# Use project venv

Before any Python-related command in this repo, activate or invoke **this** environment only:

- Activate: `.\venv\Scripts\Activate.ps1`
- Interpreter: `.\venv\Scripts\python.exe`

Do not use bare `python`, `py`, `pip`, or `pytest` from PATH. Do not use conda, system Python, or another venv.

## Workflow

From the **repository root**, in PowerShell:

```powershell
& .\venv\Scripts\Activate.ps1
python --version
```

After activation in the same shell session, `python`, `pip`, and `pytest` refer to the venv.

If activation is awkward (non-interactive, one-off calls), invoke the venv interpreter by path instead — same environment:

```powershell
& .\venv\Scripts\python.exe -m pip list
& .\venv\Scripts\python.exe -m pytest
& .\venv\Scripts\python.exe analysis\scripts\some_script.py
```

Prefer `& .\venv\Scripts\python.exe -m pip` over `pip.exe` so the correct site-packages is used.

## Linux / macOS

If the same repo is used off Windows:

```bash
source ./venv/bin/activate
# or
./venv/bin/python -m pytest
```

## If the venv is missing

1. Say that `.\venv\Scripts\Activate.ps1` (or `venv/bin/python`) was not found.
2. Ask before creating a new venv or using a different interpreter.
3. Do not silently fall back to PATH Python.
