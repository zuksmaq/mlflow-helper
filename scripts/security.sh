#!/bin/bash
# Security check with bandit

echo "🔒 Running security check with bandit..."
uv run bandit -r src/

echo "✅ Security check complete!"