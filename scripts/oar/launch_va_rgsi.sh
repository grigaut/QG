#!/bin/bash
SRCDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SCRIPT="scripts/bash/run_va_rgsi.sh"
NAME="QG3L-RGSI"
source "$SRCDIR/scripts/oar/lib.sh"

cd $SRCDIR
chmod +x $SCRIPT

parse_common_flags "$@"
load_env "$SRCDIR"


# Compute walltime
walltime=5
[ "$long_optim" = true ] && (( walltime *= 4 ))
[ "$long" = true ]       && (( walltime *= 4 ))

build_oar_opts "$walltime"

# Build base command with filtered arguments
cmd="./$SCRIPT"
for arg in "${args[@]}"; do
    cmd+=" $arg"
done

# Append extra python args based on flags
optim_args=""
if [ "$long_optim" = true ]; then
    optim_args+=" -o 800"
fi
if [ "$long" = true ]; then
    optim_args+=" --cycles 12"
fi

cmd="${cmd}${optim_args}"

oarsub "${OAR_OPTS[@]}" -n "${NAME}" "$cmd"

exit 0