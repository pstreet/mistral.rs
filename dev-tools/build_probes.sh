#!/bin/bash
# Compile the hipBLASLt C probes. Usage: ./build_probes.sh [lt_probe2|lt_run|lt_bench] [all]
set -e
ROCM=${ROCM:-/home/pstreet/LocalAI/rocm-install}
cd "$(dirname "$0")"
targets="${@:-all}"
for t in $targets; do
  [ "$t" = "all" ] && { t=lt_probe2; :; }
  if [ -f "$t.c" ]; then
    g++ -D__HIP_PLATFORM_AMD__ -O2 -o "$t" "$t.c" \
      -I"$ROCM/include" -L"$ROCM/lib" -lhipblaslt -lhipblas -lamdhip64 -lstdc++ -lm \
      -Wl,-rpath,"$ROCM/lib" 2>/dev/null
    echo "built $t"
  fi
done
# note: run with LD_LIBRARY_PATH=$ROCM/lib (rpath usually suffices)
