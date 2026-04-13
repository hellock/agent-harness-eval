# Contributing

## Setup

```bash
uv sync
```

Optional local checks:

```bash
.venv/bin/ruff check src/agent_harness_eval tests
.venv/bin/pytest tests
```

## Change Scope

- Keep changes minimal and explicit.
- Prefer improving invariants over adding fallback logic.
- Update docs in the same change when behavior or project structure changes.
- Do not commit local secrets, `eval.yaml`, `.env`, `.harnesses/`, `results/`, or cache directories.

## Tests

- Add or update tests for behavior changes.
- Keep tests isolated and filesystem-local.
- Use temporary directories instead of shared mutable fixtures.

## Pull Requests

- Use conventional commit prefixes where practical: `fix:`, `feat:`, `refactor:`, `test:`.
- Summarize behavioral impact and any config or migration changes.
- List verification performed, at minimum:
  - `.venv/bin/ruff check src/agent_harness_eval tests`
  - `.venv/bin/pytest tests`

## Issues

- Include reproduction steps, config context, and the exact failing command when reporting bugs.
- For harness-specific failures, include the affected harness, model label, and whether the run used `host` or `docker` execution.
