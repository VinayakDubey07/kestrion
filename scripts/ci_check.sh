#!/usr/bin/env bash
# scripts/ci_check.sh
# Run the full local CI check: lint → type check → tests with coverage.
# Mirrors exactly what .github/workflows/ci.yml does on every PR.
# Stops at the first failure so you see the root cause clearly.

set -euo pipefail

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Kestrion local CI check"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Activate virtual environment if present and not already activated
if [ -z "${VIRTUAL_ENV:-}" ] && [ -d ".venv" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

echo ""
echo "▶ Step 1/3 — ruff (lint + style)"
python -m ruff check src/ tests/
echo "   ✓ ruff passed"

echo ""
echo "▶ Step 2/3 — mypy (type checking)"
python -m mypy src/kestrion --ignore-missing-imports
echo "   ✓ mypy passed"

echo ""
echo "▶ Step 3/3 — pytest (tests + coverage ≥ 60%)"
python -m pytest tests/ -q
echo "   ✓ pytest passed"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " ✅ All checks passed — you're ready to open a PR!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
