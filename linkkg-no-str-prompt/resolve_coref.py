import os
import time
import argparse
import json
import requests
from datetime import datetime

host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
OLLAMA_URL = f"http://{host}/api/generate"

def log(msg, log_file):
    time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{time_stamp}] {msg}"
    print(line)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(line + "\n")

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_prompt_template(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
'''
def run_ollama_inference(prompt, model_name):
    response = requests.post(OLLAMA_URL, json={
        "model": model_name,
        "prompt": prompt,
        "stream": False
    })
    if response.status_code != 200:
        raise Exception(f"Request failed: {response.status_code}\n{response.text}")
    return response.json()['response']
'''

def run_ollama_inference(prompt, model_name, timeout=120, log_file=None):
    log_msg = f"[run_ollama_inference] Sending prompt to model '{model_name}' (len={len(prompt)}) with timeout={timeout}s"
    print(log_msg)
    if log_file:
        log(log_msg, log_file)
    
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False
            },
            timeout=timeout  # Timeout in seconds
        )
        response.raise_for_status()  # Raises HTTPError if not 200
        return response.json()['response']

    except requests.exceptions.Timeout:
        raise Exception("Ollama inference timed out.")

    except requests.exceptions.HTTPError as http_err:
        raise Exception(f"HTTP error occurred: {http_err} - {response.text}")

    except requests.exceptions.RequestException as req_err:
        raise Exception(f"Request failed: {req_err}")


def inject_prompt(template, resolved_entities, aux_descriptions, chunk_text):
    return template.strip() + "\n\n" + json.dumps({
        "RESOLVED_ENTITIES": resolved_entities,
        "AUXILIARY_DESCRIPTIONS": aux_descriptions,
        "TEXT_CHUNK": chunk_text
    }, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-dir", required=True)
    parser.add_argument("--final-memory", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--base-output-dir", default="output")
    parser.add_argument("--input-file-name", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--log-file", default=None, help="Path to the log file")
    parser.add_argument("--num-retries", type=int, default=1, help="Number of times to reprocess each chunk to improve resolution")
    parser.add_argument("--entity-type", required=True)

    args = parser.parse_args()

    #input_folder = os.path.join(args.base_output_dir, args.input_file_name)
    input_folder = args.base_output_dir 
    resolved_dir = os.path.join(input_folder, "resolved_outputs")
    os.makedirs(resolved_dir, exist_ok=True)

    log_file = os.path.join(input_folder, "log.txt")
    log("Starting coreference resolution loop", log_file)

    memory = load_json(args.final_memory)
    resolved_entities = memory.get("RESOLVED_ENTITIES", {})
    aux_descriptions = memory.get("AUXILIARY_DESCRIPTIONS", {})

    # Log what’s being removed
    removed_keys = [k for k, v in resolved_entities.items() if v is None]
    if removed_keys:
        log(f"Removing {len(removed_keys)} unresolved mappings from RESOLVED_ENTITIES: {removed_keys}", args.log_file)

    # Clean the unresolved (None) entries
    resolved_entities = {k: v for k, v in resolved_entities.items() if v is not None}

    # Replace in memory object
    memory["RESOLVED_ENTITIES"] = resolved_entities
    log(f"{len(resolved_entities)} valid RESOLVED_ENTITIES retained for final resolution.", args.log_file)
    prompt_template = load_prompt_template(args.prompt_file)

    chunk_files = sorted([f for f in os.listdir(args.chunks_dir) if f.endswith(".txt")])
    for fname in chunk_files:
        chunk_name = fname.replace(".txt", "")
        chunk_path = os.path.join(args.chunks_dir, fname)
        with open(chunk_path, 'r', encoding='utf-8') as f:
            chunk_text = f.read()

        '''
        prompt = inject_prompt(prompt_template, resolved_entities, aux_descriptions, chunk_text)

        log(f"Resolving coreference for {chunk_name}...", log_file)
        try:
            resolved_text = run_ollama_inference(prompt, args.model_name)
            out_path = os.path.join(resolved_dir, chunk_name + "_resolved.txt")
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(resolved_text)
            log(f"Saved resolved text: {out_path}", log_file)
        except Exception as e:
            log(f"Failed to process {chunk_name}: {e}", log_file)
            raise
        '''

        input_text = chunk_text
        '''
        for retry in range(args.num_retries):
            prompt = inject_prompt(prompt_template, resolved_entities, aux_descriptions, input_text)
            log(f"[Retry {retry+1}/{args.num_retries}] Resolving coreference for {chunk_name}...", log_file)
            try:
                log(f"Prompt length (chars): {len(prompt)}", log_file)
                input_text = run_ollama_inference(prompt, args.model_name, timeout=120)
            except Exception as e:
                log(f"Failed at retry {retry+1} for {chunk_name}: {e}", log_file)
                raise
        '''

        for retry in range(args.num_retries):
            prompt = inject_prompt(prompt_template, resolved_entities, aux_descriptions, input_text)
            log(f"[Retry {retry+1}/{args.num_retries}] Resolving coreference for {chunk_name}...", log_file)
            log(f"Prompt length (chars): {len(prompt)}", log_file)

            start_time = time.time()

            try:
                input_text = run_ollama_inference(prompt, args.model_name, timeout=120, log_file=args.log_file)
                elapsed = round(time.time() - start_time, 2)
                log(f"Response received in {elapsed} seconds for {chunk_name}", log_file)
                break

            except Exception as e:
                log(f"Retry {retry+1} failed for {chunk_name}: {e}", log_file)

                if "timed out" in str(e).lower():
                    log(f"Retrying {chunk_name} with extended timeout (300s)...", log_file)
                    try:
                        retry_start = time.time()
                        input_text = run_ollama_inference(prompt, args.model_name, timeout=300)
                        retry_elapsed = round(time.time() - retry_start, 2)
                        log(f"Extended timeout succeeded in {retry_elapsed} seconds for {chunk_name}", log_file)
                        break
                    except Exception as e2:
                        log(f"Extended timeout also failed at retry {retry+1} for {chunk_name}: {e2}", log_file)

                if retry == args.num_retries - 1:
                    log(f"Final Resolution failed for {chunk_name}.", log_file)
                    with open(f"debug_failed_prompt_{chunk_name}.txt", "w", encoding="utf-8") as f:
                        f.write(prompt)
                    raise

        resolved_text = input_text
        out_path = os.path.join(resolved_dir, chunk_name + "_resolved.txt")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(resolved_text)
        log(f"Saved resolved text: {out_path}", log_file)

    # === Merge all resolved chunks into one final file ===
    merged_path = os.path.join(input_folder, f"{args.entity_type}_resolved_{args.input_file_name}.txt")
    log(f"Merging all resolved chunks into {merged_path}", log_file)

    resolved_files = sorted([
        f for f in os.listdir(resolved_dir)
        if f.endswith("_resolved.txt")
    ])

    with open(merged_path, 'w', encoding='utf-8') as merged_file:
        for fname in resolved_files:
            fpath = os.path.join(resolved_dir, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                merged_file.write(f.read().strip() + "\n\n")

    log(f"All resolved chunks merged and saved to {merged_path}", log_file)

    log("Completed coreference resolution for all chunks.", log_file)

if __name__ == "__main__":
    main()
