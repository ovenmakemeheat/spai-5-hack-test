#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 32
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 03:00:00
#SBATCH -A zz992004 
#SBATCH -J jupyter
# SBATCH --nodelist=lanta-g-004

port=$(shuf -i 6000-9999 -n 1)
USER=$(whoami)
node=$(hostname -s)

ml load Miniforge3/25.3.0-3 cuda/12.6
conda activate demo

# jupyter notebookng instructions to the output file
echo -e "
Jupyter server is running on: $(hostname)
Job starts at: $(date)
Copy/Paste the following command into your local terminal
--------------------------------------------------------------------
ssh -L $port:$node:$port $USER@lanta.nstda.or.th
--------------------------------------------------------------------
Open a browser on your local machine with the following address
--------------------------------------------------------------------
http://localhost:${port}/?token=XXXXXXXX (see your token below)
--------------------------------------------------------------------
“
"

# start a cluster instance and launch the jupyter server
unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi
jupyter notebook --no-browser --port $port --notebook-dir=$(pwd) --ip=$node