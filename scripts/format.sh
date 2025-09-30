#!/bin/bash
# Format code with black and isort

echo "🎨 Formatting code with black..."
uv run black src/ examples/

echo "📦 Sorting imports with isort..."
uv run isort src/ examples/

echo "✅ Code formatting complete!"