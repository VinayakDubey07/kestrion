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

## Running the full CI check locally

Before opening a PR, run **all three checks** in this order — the same checks the CI pipeline runs:

```bash
# 1. Lint + style (zero warnings expected)
ruff check src/ tests/

# 2. Type checking
mypy src/kestrion --ignore-missing-imports

# 3. Tests with coverage (must stay ≥ 60%)
pytest tests/ -v
```

Or use the convenience script (runs all three, stops on first failure):

```bash
./scripts/ci_check.sh
```

All three run automatically on every push and PR via `.github/workflows/ci.yml`
across Python 3.10–3.12.

### Coverage reports

After running `pytest`, open `htmlcov/index.html` in your browser for a line-level
coverage view. The CI gate is **60% minimum** — pull requests that drop below this
will fail.

### Type checking

Kestrion ships a `py.typed` marker (PEP 561) so downstream users get full
type-checking support. All new public APIs must be fully annotated. Run:

```bash
mypy src/kestrion --ignore-missing-imports --strict
```

## Guidelines

* **Keep PRs small and scoped.** One concept per PR (e.g. "fix Ollama malformed
  tool-call JSON," not "fix Ollama + add scheduler").
* **Add or update tests** for any behavior change. New code under
  `src/kestrion/` should have a corresponding test under `tests/`.
* **Match the existing style.** `ruff` enforces formatting/lint; there's no
  separate style guide to memorize.
* **If you're making major changes to core components** (like the scheduler, CLI, telemetry providers, or persistence engines), open an issue first to align on the technical approach before writing a large PR.
* **Update `CHANGELOG.md`** under an `## [Unreleased]` heading for any
  user-facing change.

## Deprecating old APIs

When renaming or removing a public API, use the `@deprecated` decorator rather
than deleting it immediately. This gives users at least one minor version to migrate:

```python
from kestrion.core.compat import deprecated

@deprecated(since="0.8.0", use="NewClassName")
class OldClassName:
    ...
```

The decorator emits a `DeprecationWarning` on first call and updates the docstring
automatically. Remove the shim one minor version after the deprecation is published.

## Reporting bugs

Open a GitHub issue with: what you ran, what you expected, what happened
instead, and your Python version. A minimal reproduction (a short script,
not your whole agent) gets fixed fastest.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).