#!/bin/bash
SRCDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd $SRCDIR

# RGSI

./scripts/oar/launch_va_enatl60_rgsi.sh --config=config/va_enatl60_rgsi_summer.toml -v "$@"
./scripts/oar/launch_va_enatl60_rgsi.sh --config=config/va_enatl60_rgsi_winter.toml -v "$@"

# Forced

./scripts/oar/launch_va_enatl60_forced.sh --config=config/va_enatl60_forced_summer.toml -v "$@"
./scripts/oar/launch_va_enatl60_forced.sh --config=config/va_enatl60_forced_winter.toml -v "$@"