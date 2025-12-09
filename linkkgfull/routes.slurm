#!/bin/bash
#SBATCH --partition=contrib-gpuq                    # the DGX only belongs in the 'gpu'  partition
#SBATCH --qos=cs_dept                          # need to select 'gpu' QoS
#SBATCH --job-name=routesrunpipeline-llmlink
#SBATCH --output=/scratch/dmeher/slurm_outputs/routesrunpipeline-llmlink.%j.out
#SBATCH --error=/scratch/dmeher/slurm_outputs/routesrunpipeline-llmlink.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                 # up to 128; 
#SBATCH --gres=gpu:A100.80gb:1
##SBATCH --mem-per-cpu=160GB                 # memory per CORE; total memory is 1 TB (1,000,000 MB)
#SBATCH --cpus-per-task=1 
#SBATCH --mem=30GB
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

# Start Ollama Server
#/projects/cdomenic/HS_CINA/Dipak/ollama/ollama serve &

# === Assign a dynamic port === 
export OLLAMA_PORT=$(( ( RANDOM % (11835 - 11435 + 1) ) + 11435 ))
export OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"

# === Prevent Ollama from shutting down automatically ===
export OLLAMA_KEEP_ALIVE=-1

# ✅ Set the full API base (just reuse OLLAMA_HOST)
export OLLAMA_API_BASE="http://${OLLAMA_HOST}/v1"

# === Start Ollama and capture PID ===
ollama serve &
OLLAMA_PID=$!

# Setup trap to kill Ollama when the job exits
trap "kill $OLLAMA_PID" EXIT

# Wait a few seconds for Ollama to boot
sleep 5

# === YOUR ACTUAL JOB ===
echo "[$(date)] [$$] Running NER job..." >> "$LOGFILE"


echo "Current directory: $(pwd)"

#python run_pipeline.py \
 # --input-file ./input/02smallyusuf.txt \
  #--max-tokens 300 \
  #--min-last-chunk-words 50 \
  #--use-tokenizer

#python run_pipeline1.py \
 # --input-file ./input/02smallyusuf.txt \
  #--max-tokens 300\
  #--min-last-chunk-words 20 \
  #--use-tokenizer \
  #--ner-prompt-file ./prompts/person_nopr_ner_prompt.txt \
  #--ner-model-name llama3370gb32k  \
  #--ner-max-retries 3

#python run_pipeline2.py \
 # --input-file ./input/02smallyusuf.txt \
  #--max-tokens 300 \
  #--min-last-chunk-words 20 \
  #--use-tokenizer \
  #--ner-prompt-file prompts/person_nopr_ner_prompt.txt \
  #--ner-model-name llama3370gb32k \
  #--ner-max-retries 3 \
  #--coref-prompt-file prompts/person_nopr_coref_prompt.txt \
  #--coref-verify-prompt-file prompts/person_verify_prompt.txt \
  #--coref-model-name lliama3370gb32k \
  #--coref-verify-passs 3

#python run_pipeline4.py \
 # --input-file ./input/20USVsGarcia.txt \
  #--max-tokens 300 \
  #--min-last-chunk-words 20 \
  #--use-tokenizer \
  #--ner-prompt-file prompts/person_nopr_ner_prompt.txt \
  #--ner-model-name llama3370gb32k \
  #--ner-max-retries 3 \
  #--coref-prompt-file prompts/person_nopr_coref_prompt.txt \
  #--coref-verify-prompt-file prompts/person_verify_prompt.txt \
  #--coref-model-name llama3370gb32k \
  #--coref-verify-passes 2 \
  #--resolve-prompt-file prompts/person_nopr_resolve_prompt.txt \
  #--resolve-model-name llama3370gb32k 

python run_pipeline5.py \
  --input-file ./input/20USVsGarcia.txt \
  --input-file-name 20USVsGarcia \
  --entity-type routes \
  --max-tokens 300 \
  --min-last-chunk-words 50 \
  --use-tokenizer \
  --ner-prompt-file prompts/routes_nopr_ner_prompt.txt \
  --ner-model-name llama3370gb32k \
  --ner-max-retries 3 \
  --coref-prompt-file prompts/routes_nopr_coref_prompt.txt \
  --coref-verify-prompt-file prompts/routes_nopr_coref_prompt.txt \
  --coref-model-name llama3370gb32k \
  --coref-verify-passes 2 \
  --coref-max-retries 3 \
  --resolve-prompt-file prompts/routes_nopr_resolve_prompt.txt \
  --resolve-model-name llama3370gb32k \
  --resolve-num-retries 3 \
  --run-stages chunk ner coref resolve  #chunk ner coref resolve

echo "[$(date)] [$$] NER job finished." >> "$LOGFILE"


