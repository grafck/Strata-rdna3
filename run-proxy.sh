#!/bin/sh
cd "/home/graf2/dev/inference/Strata-rdna3" || exit 1
exec .venv/bin/python proxy.py "$@"
