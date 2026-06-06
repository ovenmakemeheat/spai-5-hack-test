# Claude Instructions

## Package Management

- **Do NOT** run `pip`, `pip3`, or any pip-related commands.
- When adding Python packages, always use:

```powershell
uv add {package} --no-sync
```

- Do not run `uv add` without `--no-sync`.

## Python Runtime

This project uses [uv](https://github.com/astral-sh/uv). To run Python:

- Preferred: `uv run python <script>`
- Alternative: `.venv\Scripts\python.exe <script>`

## HPC Environment
- you can use model or scale method based on HPC environment with 40gb vram