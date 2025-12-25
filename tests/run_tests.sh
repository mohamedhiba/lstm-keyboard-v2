#!/bin/bash

# Milestone 2 Test Runner
# This script runs all shape and training mechanics tests

echo "================================="
echo "  MILESTONE 2: Shape Tests"
echo "================================="
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "❌ pytest not found. Installing..."
    pip install pytest
    echo ""
fi

# Run tests with verbose output
echo "Running tests..."
echo ""

pytest tests/test_shapes.py -v --tb=short

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "================================="
    echo "  ✅ MILESTONE 2 COMPLETE!"
    echo "================================="
    echo ""
    echo "All tests passed! Your model:"
    echo "  ✓ Produces correct output shapes"
    echo "  ✓ Can compute loss without errors"
    echo "  ✓ Has working backward pass"
    echo ""
    echo "Next step: Implement training loop!"
else
    echo ""
    echo "================================="
    echo "  ❌ TESTS FAILED"
    echo "================================="
    echo ""
    echo "Please fix the errors above before proceeding."
    exit 1
fi