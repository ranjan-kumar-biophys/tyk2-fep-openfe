#!/bin/bash
# Submit the FEP job once another job has left the queue (gpuq allows 48 cores per user,
# so it cannot be queued behind it) and the conda env has built. Run with nohup on the
# login node:  nohup bash hpc/submit_after.sh <job id> > submit_after.log 2>&1 &
cd "$(dirname "$0")/.."
while qstat "$1" > /dev/null 2>&1; do sleep 300; done
until grep -q "^ENV_EXIT=" env_install.log 2>/dev/null; do sleep 60; done
grep -q "^ENV_EXIT=0" env_install.log || { echo "env build failed $(date)" > SUBMIT_FAILED; exit 1; }
qsub hpc/run_pbs_gpu02.pbs > submitted.txt 2>&1 || cp submitted.txt SUBMIT_FAILED
