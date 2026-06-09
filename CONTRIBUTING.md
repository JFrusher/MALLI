# Contributing to MALLI

Thanks for helping improve MALLI.

## Workflow

- Keep changes focused and scoped to a single behavior or module.
- Update docs when behavior, commands, or file locations change.
- Prefer the existing Python and Flutter conventions in this repo.
- Avoid checking in generated artifacts, models, caches, or datasets.

## Before opening a pull request

- Run the relevant Python or Flutter checks for the files you changed.
- Verify the README or supporting docs still match the code.
- Add or update tests when you change logic in `models/`, `data/`, or `lib/`.

## Where to start

- Training and export: `train.py`, `models/`, `data/`
- Mobile app: `lib/`
- Documentation: `docs/`

If you are unsure where a change belongs, start by checking the nearest module doc or implementation summary and keep the change small.