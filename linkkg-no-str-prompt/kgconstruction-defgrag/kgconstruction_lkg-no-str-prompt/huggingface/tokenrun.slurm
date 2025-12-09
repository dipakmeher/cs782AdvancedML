#!/bin/bash
#SBATCH --partition=contrib-gpuq                    # the DGX only belongs in the 'gpu'  partition
#SBATCH --qos=cs_dept                          # need to select 'gpu' QoS
#SBATCH --job-name=countgraphrag
#SBATCH --output=/scratch/dmeher/slurm_outputs/countgraphrag.%j.out
#SBATCH --error=/scratch/dmeher/slurm_outputs/countgraphrag.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                 # up to 128; 
#SBATCH --gres=gpu:A100.80gb:1
##SBATCH --mem-per-cpu=160GB                 # memory per CORE; total memory is 1 TB (1,000,000 MB)
#SBATCH --cpus-per-task=1 
#SBATCH --mem=10GB
#SBATCH --export=ALL
#SBATCH --time=01-00:00:00                   # set to 1hr; please choose carefully
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
source /scratch/dmeher/custom_env/miniforge/bin/activate
conda activate graphrag_env032



# Run GraphRAG script
/scratch/dmeher/custom_env/miniforge/envs/graphrag_env032/bin/python main.py --input ../forloop_coreference_resolution/noncorinput/04USVsRodriguez.txt --prompt   
#/projects/cdomenic/HS_CINA/Dipak/ollama/ollama serve & 
#/scratch/dmeher/custom_env/miniforge/envs/graphrag_env032/bin/python /scratch/dmeher/GraphRAGProject/project7_graphrag032_monkeypatch/index.py --root /scratch/dmeher/GraphRAGProject/project7_graphrag032_monkeypatch/ragtest 
