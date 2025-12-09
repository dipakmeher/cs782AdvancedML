import os
import argparse
import json
import requests
from datetime import datetime
import time

def log(msg, log_file_path=None):
    time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{time_stamp}] {msg}"
    print(full_msg)
    if log_file_path:
        with open(log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write(full_msg + "\n")

def extract_json_from_ollama(raw_response):
    cleaned = raw_response.strip()
    if not cleaned.startswith("{"):
        raise ValueError(f"Output does not start with '{{'. Unexpected format:\n{cleaned[:200]}")

    try:
        result = json.loads(cleaned)
        if "RESOLVED_ENTITIES" not in result or "AUXILIARY_DESCRIPTIONS" not in result:
            raise ValueError("Valid JSON but missing required keys.")
        return result
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format. {e}")

def load_prompt_template(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
'''
def run_ollama_inference(prompt, model, url):
    response = requests.post(url, json={
        "model": model,
        "prompt": prompt,
        "stream": False
    })
    if response.status_code != 200:
        raise Exception(f"Request failed: {response.status_code}\n{response.text}")
    return response.json()['response']
'''

def run_ollama_inference(prompt, model, url, timeout=120):
    try:
        response = requests.post(
            url,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=timeout  # Set timeout in seconds
        )
        response.raise_for_status()  # Raises an error for 4xx/5xx responses
        return response.json()['response']

    except requests.exceptions.Timeout:
        raise Exception("Ollama inference timed out.")

    except requests.exceptions.HTTPError as http_err:
        raise Exception(f"HTTP error occurred: {http_err} - {response.text}")

    except requests.exceptions.RequestException as req_err:
        raise Exception(f"Request failed: {req_err}")


def inject_prompt(prompt_template, resolved_entities, aux_descriptions, ner_entities, aux_current, chunk_text, verification=False):
    identified_entities = {
        "NER_ENTITIES": ner_entities,
        "PROPER_NOUN_DESCRIPTION": aux_current
    }

    payload = {
        "RESOLVED_ENTITIES": resolved_entities,
        "AUXILIARY_DESCRIPTIONS": aux_descriptions,
        "IDENTIFIED_ENTITIES": identified_entities,
        "CHUNK_TEXT": chunk_text
    }

    return prompt_template.strip() + "\n\n" + json.dumps(payload, indent=2)

def process_chunks(chunk_files, args, prompt_template, resolved_entities, aux_descriptions, verification=False):
    for fname in chunk_files:
        if not fname.endswith(".txt"):
            continue

        chunk_name = fname.replace(".txt", "")
        chunk_path = os.path.join(args.chunks_dir, fname)
        ner_path = os.path.join(args.ner_dir, chunk_name + ".json")

        with open(chunk_path, 'r', encoding='utf-8') as f:
            chunk_text = f.read()
        with open(ner_path, 'r', encoding='utf-8') as f:
            ner_data = json.load(f)

        ner_entities = ner_data.get("ENTITIES", {})
        aux_current = ner_data.get("PROPER_NOUN_DESCRIPTION", {})

        full_prompt = inject_prompt(
            prompt_template,
            resolved_entities,
            aux_descriptions,
            ner_entities,
            aux_current,
            chunk_text,
            verification=verification
        )
        log(f"Processing {chunk_name} (verification={verification})...", args.log_file)
        
        '''
        try:
            result_raw = run_ollama_inference(full_prompt, args.model, args.ollama_url)
            result = extract_json_from_ollama(result_raw)
        except Exception as e:
            log(f"Failed to process {chunk_name}: {e}", args.log_file)
            raise
        '''
       
        '''
        max_retries = max(1, args.max_retries if hasattr(args, "max_retries") else 3)
        attempt = 0
        success = False
        while attempt < max_retries and not success:
            try:
                #result_raw = run_ollama_inference(full_prompt, args.model, args.ollama_url)
                log(f"Prompt length (chars): {len(full_prompt)}", args.log_file)
                result_raw = run_ollama_inference(full_prompt, args.model, args.ollama_url, timeout=120)            
                try:
                    result = extract_json_from_ollama(result_raw)
                    success = True
                except Exception as json_error:
                    log(f"[Attempt {attempt+1}] JSON parsing failed for {chunk_name}. Raw model output:\n{result_raw}", args.log_file)
                    raise json_error  # Let it go to outer except to handle retries
            except Exception as e:
                attempt += 1
                log(f"[Attempt {attempt}] Failed to process {chunk_name}: {e}", args.log_file)
                if attempt == max_retries:
                    log(f"Maximum retries reached for {chunk_name}. Skipping.", args.log_file)
                    raise
        '''

        max_retries = max(1, args.max_retries if hasattr(args, "max_retries") else 3)
        timeout_retry_done = False
        attempt = 0
        success = False
        last_error = None

        while attempt < max_retries and not success:
            try:
                log(f"Prompt length (chars): {len(full_prompt)}", args.log_file)
                start_time = time.time()
                result_raw = run_ollama_inference(full_prompt, args.model, args.ollama_url, timeout=120)
                elapsed = round(time.time() - start_time, 2)

                log(f"[Attempt {attempt + 1}] Response received in {elapsed} seconds for {chunk_name}", args.log_file)

                result = extract_json_from_ollama(result_raw)
                success = True

            except Exception as e:
                error_message = str(e).lower()
                last_error = e
                log(f"[Attempt {attempt + 1}] Failed to process {chunk_name}: {e}", args.log_file)
                attempt += 1

                # Retry once with extended timeout if it's a timeout error
                if "timed out" in error_message and not timeout_retry_done:
                    log("Timeout detected. Retrying once with increased timeout (300 seconds).", args.log_file)
                    timeout_retry_done = True
                    try:
                        retry_start = time.time()
                        result_raw = run_ollama_inference(full_prompt, args.model, args.ollama_url, timeout=300)
                        retry_elapsed = round(time.time() - retry_start, 2)

                        log(f"[Extended Timeout] Response received in {retry_elapsed} seconds for {chunk_name}", args.log_file)

                        result = extract_json_from_ollama(result_raw)
                        success = True
                        continue  # Skip the rest and start the next iteration (which won't happen because success=True)
                    except Exception as retry_e:
                        retry_elapsed = round(time.time() - retry_start, 2)
                        last_error = retry_e
                        log(f"[Extended Timeout] Failed after {retry_elapsed} seconds for {chunk_name}: {retry_e}", args.log_file)
                        log(f"Retry after timeout also failed: {retry_e}", args.log_file)
                        if not success:
                            attempt += 1

        if not success:
            log(f"Maximum retries reached for {chunk_name}. Skipping.", args.log_file)
            raise Exception(f"Ollama inference failed for {chunk_name} after {attempt} attempts. Last error: {last_error}")

        resolved_entities.update(result.get("RESOLVED_ENTITIES", {}))
        aux_descriptions.update(result.get("AUXILIARY_DESCRIPTIONS", {}))

        out_path = os.path.join(args.output_dir, chunk_name + ("_verify" if verification else "") + ".json")
        with open(out_path, 'w', encoding='utf-8') as out_file:
            json.dump(result, out_file, indent=2)

        log(f"Saved coref output: {out_path}", args.log_file)

    final_memory_path = os.path.join(args.base_output_folder, "final_memory.json")
    with open(final_memory_path, 'w', encoding='utf-8') as f:
        json.dump({
            "RESOLVED_ENTITIES": resolved_entities,
            "AUXILIARY_DESCRIPTIONS": aux_descriptions
        }, f, indent=2)
    log(f"Final memory saved to: {final_memory_path}", args.log_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-dir", required=True)
    parser.add_argument("--ner-dir", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--verify-prompt-file")
    parser.add_argument("--base-output-folder", required=True)
    parser.add_argument("--input-file-name", required=True)
    parser.add_argument("--model", default="llama3370gb32k")
    parser.add_argument("--verify-passes", type=int, default=0)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries for failed coref requests")
    args = parser.parse_args()

    # Set Ollama URL here
    args.ollama_url = f"http://{os.environ.get('OLLAMA_HOST', '127.0.0.1:11434')}/api/generate"

    output_path = os.path.join(args.base_output_folder, "coref_outputs")
    os.makedirs(output_path, exist_ok=True)
    args.output_dir = output_path

    discovery_prompt = load_prompt_template(args.prompt_file)
    verify_prompt = load_prompt_template(args.verify_prompt_file) if args.verify_prompt_file else discovery_prompt
    chunk_files = sorted(os.listdir(args.chunks_dir))

    resolved_entities = {}
    aux_descriptions = {}

    process_chunks(chunk_files, args, discovery_prompt, resolved_entities, aux_descriptions, verification=False)

    for i in range(args.verify_passes):
        log(f"Starting verification pass {i+1}...", args.log_file)
        process_chunks(chunk_files, args, verify_prompt, resolved_entities, aux_descriptions, verification=True)

if __name__ == "__main__":
    main()
