#!/bin/bash
SRCDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd $SRCDIR

# RGSI

./scripts/oar/launch_va_rgsi.sh --config=config/va_rgsi_z2.toml -v "$@"
./scripts/oar/launch_va_rgsi.sh --config=config/va_rgsi_z3.toml -v "$@"

# Forced

./scripts/oar/launch_va_forced.sh --config=config/va_forced_z2.toml -v "$@"
./scripts/oar/launch_va_forced.sh --config=config/va_forced_z3.toml -v "$@"