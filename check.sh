#!/bin/bash

# Code quality checks for the RAG system

set -e

echo "Running code quality checks..."

echo ""
echo "==> black (formatting check)"
uv run black backend/ main.py --check

echo ""
echo "All checks passed!"
