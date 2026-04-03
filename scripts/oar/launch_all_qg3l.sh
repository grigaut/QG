#!/bin/bash
SRCDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd $SRCDIR

# SurfML

./scripts/oar/launch_va_surfml.sh --config=config/va_surfml_z2.toml -v "$@"
./scripts/oar/launch_va_surfml.sh --config=config/va_surfml_z3.toml -v "$@"

# Forced

./scripts/oar/launch_va_forced.sh --config=config/va_forced_z2.toml -v "$@"
./scripts/oar/launch_va_forced.sh --config=config/va_forced_z3.toml -v "$@"