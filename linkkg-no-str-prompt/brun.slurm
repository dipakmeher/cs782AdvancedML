#!/bin/bash
#SBATCH --partition=contrib-gpuq                    # the DGX only belongs in the 'gpu'  partition
#SBATCH --qos=cs_dept                          # need to select 'gpu' QoS
#SBATCH --job-name=length
#SBATCH --output=/projects/cdomenic/HS_CINA/Dipak/slurm_outputs/length.%j.out
#SBATCH --error=/projects/cdomenic/HS_CINA/Dipak/slurm_outputs/length.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                 # up to 128; 
#SBATCH --gres=gpu:1g.10gb:1
##SBATCH --mem-per-cpu=160GB                 # memory per CORE; total memory is 1 TB (1,000,000 MB)
#SBATCH --cpus-per-task=1 
#SBATCH --mem=10GB
#SBATCH --export=ALL
#SBATCH --time=02-00:00:00                   # set to 1hr; please choose carefully
#SBATCH --mail-type=BEGIN,END,FAIL     # NONE,BEGIN,END,FAIL,REQUEUE,ALL,...
#SBATCH --mail-user=dmeher@gmu.edu   # Put your GMU email address here

set echo
umask 0027

# to see ID and state of GPUs assigned
nvidia-smi

#source /scratch/dmeher/custom_env/recguru_env/bin/activate
#module load ollama
#module load gnu10
#module load python
# Load Miniforge (only necessary if Conda is not initialized globally)
#source /scratch/dmeher/custom_env/miniforge/etc/profile.d/conda.sh
#source /projects/cdomenic/HS_CINA/Dipak/custom_env/miniforge/bin/activate
#conda activate graphrag_env032

module purge
module load gnu10
module load python/3.10

source /projects/cdomenic/HS_CINA/Dipak/GraphRAGProject/custom_env/graphrag_env/bin/activate

# Start Ollama Server
#/projects/cdomenic/HS_CINA/Dipak/ollama/ollama serve &

python ./input/length1.py
