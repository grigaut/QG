#!/bin/bash

lscpu | grep 'Model name' | cut -f 2 -d ":" | awk '{$1=$1}1'

echo JOB ID : $OAR_JOB_ID

SRCDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

cd $SRCDIR

date

LD_PRELOAD=./.venv/lib/libstdc++.so.6 .venv/bin/python3 -u scripts/va_enatl60_surfml.py $@

date

exit 0