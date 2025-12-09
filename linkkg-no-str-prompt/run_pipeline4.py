import argparse
import subprocess
import os
import time
from datetime import datetime

def log(message, log_file=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {message}"
    print(full_msg)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(full_msg + '\n')

def run_chunk_stage(args, input_file_name, output_dir, log_file):
    log("Starting chunking step...", log_file)
    start = time.time()

    chunk_output_dir = os.path.join(output_dir, "chunk_outputs")
    os.makedirs(chunk_output_dir, exist_ok=True)

    chunk_cmd = [
        "python", "chunk.py",
        "--input-file", args.input_file,
        "--output-dir", chunk_output_dir,
        "--max-tokens", str(args.max_tokens),
        "--min-last-chunk-words", str(args.min_last_chunk_words)
    ]
    if args.use_tokenizer:
        chunk_cmd.append("--use-tokenizer")

    result = subprocess.run(chunk_cmd)
    end = time.time()

    if result.returncode == 0:
        log(f"Chunking completed successfully in {end - start:.2f} seconds.", log_file)
    else:
        log("Chunking failed. Error details:", log_file)
        exit(1)

def run_ner_stage(args, chunk_output_dir, output_dir, log_file):
    log("Starting NER step...", log_file)
    start = time.time()

    ner_output_dir = os.path.join(output_dir, "ner_outputs")
    os.makedirs(ner_output_dir, exist_ok=True)

    ner_cmd = [
        "python", "ner.py",
        "--chunks-dir", chunk_output_dir,
        "--prompt-file", args.ner_prompt_file,
        "--output-dir", ner_output_dir,
        "--log-file", log_file,
        "--model-name", args.ner_model_name,
        "--max-retries", str(args.ner_max_retries)
    ]

    result = subprocess.run(ner_cmd)
    end = time.time()

    if result.returncode == 0:
        log(f"NER completed successfully in {end - start:.2f} seconds.", log_file)
    else:
        log("NER failed. Error details:", log_file)
        exit(1)

def run_coref_stage(args, chunk_output_dir, ner_output_dir, output_dir, input_file_name, log_file):
    log("Starting Coreference Resolution step...", log_file)
    start = time.time()

    coref_cmd = [
        "python", "loopcoref.py",
        "--chunks-dir", chunk_output_dir,
        "--ner-dir", ner_output_dir,
        "--prompt-file", args.coref_prompt_file,
        "--base-output-folder", "output",
        "--input-file-name", input_file_name,
        "--model", args.coref_model_name,
        "--verify-passes", str(args.coref_verify_passes),
        "--log-file", log_file
    ]

    if args.coref_verify_prompt_file:
        coref_cmd.extend(["--verify-prompt-file", args.coref_verify_prompt_file])

    result = subprocess.run(coref_cmd)
    end = time.time()

    if result.returncode == 0:
        log(f"Coreference Resolution completed successfully in {end - start:.2f} seconds.", log_file)
    else:
        log("Coreference Resolution failed. Error details:", log_file)
        exit(1)

def run_resolve_stage(args, chunk_output_dir, output_dir, input_file_name, log_file):
    log("Starting Final Coref Resolution step...", log_file)
    start = time.time()

    resolved_output_dir = os.path.join(output_dir, "resolved_outputs")
    os.makedirs(resolved_output_dir, exist_ok=True)

    final_memory_path = os.path.join(output_dir, "final_memory.json")

    resolve_cmd = [
        "python", "resolve_coref.py",
        "--chunks-dir", chunk_output_dir,
        "--final-memory", final_memory_path,
        "--prompt-file", args.resolve_prompt_file,
        "--base-output-dir", "output",
        "--input-file-name", input_file_name,
        "--model-name", args.resolve_model_name
    ]

    result = subprocess.run(resolve_cmd)
    end = time.time()

    if result.returncode == 0:
        log(f"Final Coref Resolution completed successfully in {end - start:.2f} seconds.", log_file)
    else:
        log("Final Coref Resolution failed. Error details:", log_file)
        exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run full pipeline on legal text.")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--min-last-chunk-words", type=int, default=20)
    parser.add_argument("--use-tokenizer", action="store_true")

    # NER args
    parser.add_argument("--ner-prompt-file", required=True)
    parser.add_argument("--ner-model-name", required=True)
    parser.add_argument("--ner-max-retries", type=int, default=2)

    # Coref args
    parser.add_argument("--coref-prompt-file", required=True)
    parser.add_argument("--coref-verify-prompt-file")
    parser.add_argument("--coref-model-name", required=True)
    parser.add_argument("--coref-verify-passes", type=int, default=0)

    # Final Resolution args
    parser.add_argument("--resolve-prompt-file", required=True)
    parser.add_argument("--resolve-model-name", required=True)
    
    args = parser.parse_args()

    input_file_name = os.path.splitext(os.path.basename(args.input_file))[0]
    output_dir = os.path.join("output", input_file_name)
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "log.txt")
    log(f"Pipeline started for {input_file_name}", log_file)

    run_chunk_stage(args, input_file_name, output_dir, log_file)

    chunk_output_dir = os.path.join(output_dir, "chunk_outputs")
    run_ner_stage(args, chunk_output_dir, output_dir, log_file)

    ner_output_dir = os.path.join(output_dir, "ner_outputs")
    run_coref_stage(args, chunk_output_dir, ner_output_dir, output_dir, input_file_name, log_file)

    run_resolve_stage(args, chunk_output_dir, output_dir, input_file_name, log_file)

    log("✅ Full pipeline completed: chunking → NER → coref → resolve", log_file)

if __name__ == "__main__":
    main()
