#!/bin/bash

# Step 1: Parse --filename argument
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --filename) FILENAME="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; echo "Usage: ./submit_pipeline_job.sh --filename YourFileName"; exit 1 ;;
    esac
    shift
done

if [ -z "$FILENAME" ]; then
    echo "ERROR: Missing --filename argument"
    echo "Usage: ./submit_pipeline_job.sh --filename YourFileName"
    exit 1
fi

SLURM_JOB_FILE="job_${FILENAME}.slurm"

# Step 2: Write SLURM script dynamically
cat <<EOF > $SLURM_JOB_FILE
#!/bin/bash
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --job-name=$FILENAME-runpipeline
#SBATCH --output=/scratch/dmeher/slurm_outputs/$FILENAME-runpipeline.%j.out
#SBATCH --error=/scratch/dmeher/slurm_outputs/$FILENAME-runpipeline.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100GB
#SBATCH --time=05-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dmeher@gmu.edu

set -e
set -o pipefail
umask 0027
nvidia-smi

source /scratch/dmeher/custom_env/miniforge/bin/activate
conda activate graphrag_env032

export OLLAMA_PORT=\$(( ( RANDOM % (11835 - 11435 + 1) ) + 11435 ))
export OLLAMA_HOST="127.0.0.1:\$OLLAMA_PORT"
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_API_BASE="http://\${OLLAMA_HOST}/v1"

ollama serve &
OLLAMA_PID=\$!
trap "kill \$OLLAMA_PID" EXIT
sleep 5

echo "[\$(date '+%Y-%m-%d %H:%M:%S')] Starting PERSON  pipeline for \${FILENAME}"

python run_pipeline5.py \\
  --input-file ./input/${FILENAME}.txt \\
  --input-file-name ${FILENAME} \\
  --entity-type person \\
  --max-tokens 300 \\
  --min-last-chunk-words 50 \\
  --use-tokenizer \\
  --ner-prompt-file prompts/person_nopr_ner_prompt.txt \\
  --ner-model-name llama3370gb32k \\
  --ner-max-retries 3 \\
  --coref-prompt-file prompts/person_nopr_coref_prompt.txt \\
  --coref-verify-prompt-file prompts/person_nopr_coref_prompt.txt \\
  --coref-model-name llama3370gb32k \\
  --coref-verify-passes 2 \\
  --coref-max-retries 3 \\
  --resolve-prompt-file prompts/person_nopr_resolve_prompt.txt \\
  --resolve-model-name llama3370gb32k \\
  --resolve-num-retries 3 \\
  --run-stages resolve

EOF

# Step 3: Submit the job
sbatch $SLURM_JOB_FILE

# Optional cleanup
# rm $SLURM_JOB_FILE
