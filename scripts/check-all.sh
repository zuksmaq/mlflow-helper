#!/bin/bash
# Run all code quality checks

echo "🚀 Running all code quality checks..."
echo

./scripts/format.sh
echo

./scripts/lint.sh
echo

./scripts/typecheck.sh
echo

./scripts/security.sh
echo

echo "🎉 All checks complete!"