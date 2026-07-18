# Contributing to Kestrion

Thanks for considering a contribution. Kestrion is designed to be a robust, enterprise-grade framework, so we highly value contributions that improve stability, security, and scalability. Bug reports, small fixes, and design discussions on roadmap items are always welcome.

## Setup

```bash
git clone https://github.com/VinayakDubey07/kestrion.git
cd kestrion
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Before opening a PR

Run the same checks CI runs:

```bash
ruff check src/ tests/
mypy src/kestrion --ignore-missing-imports
pytest tests/ -v
```

All three run automatically on every push and PR via `.github/workflows/ci.yml`
across Python 3.10–3.12.

## Guidelines

* **Keep PRs small and scoped.** One concept per PR (e.g. "fix Ollama malformed
  tool-call JSON," not "fix Ollama + add scheduler").
* **Add or update tests** for any behavior change. New code under
  `src/kestrion/` should have a corresponding test under `tests/`.
* **Match the existing style.** `ruff` enforces formatting/lint; there's no
  separate style guide to memorize.
* **If you're making major changes to core components** (like the scheduler, CLI, telemetry providers, or persistence engines), open an issue first to align on the technical approach before writing a large PR.
* **Update `CHANGELOG.md`** under an "Unreleased" heading for any
  user-facing change.

## Reporting bugs

Open a GitHub issue with: what you ran, what you expected, what happened
instead, and your Python version. A minimal reproduction (a short script,
not your whole agent) gets fixed fastest.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).