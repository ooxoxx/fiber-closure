#!/bin/bash
set -e

# Add darknet python bindings to path if available
if [ -d "/app/darknet/src-python" ]; then
    export PYTHONPATH="/app/darknet/src-python:$PYTHONPATH"
fi

# Verify darknet library exists
if [ ! -f "$FIBER_DARKNET_LIB_PATH" ]; then
    echo "Warning: Darknet library not found at $FIBER_DARKNET_LIB_PATH"
    echo "Please mount the library: -v /path/to/libdarknet.so:/app/lib/libdarknet.so"
fi

exec "$@"
