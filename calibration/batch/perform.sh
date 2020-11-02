#!/bin/sh

# CD into our working directory
cd ${DIRNAME}

# Run on a single node
python -c "import batch; batch.perform($SLURM_ARRAY_TASK_ID)"