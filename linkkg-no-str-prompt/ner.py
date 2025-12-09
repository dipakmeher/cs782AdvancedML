import os
import argparse
import json
import requests
from datetime import datetime
from tqdm import tqdm
import re

OLLAMA_URL_TEMPLATE = "http://{host}/api/generate"


def log(msg, log_file=None):
    time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{time_stamp}] {msg}"
    print(formatted)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(formatted + '\n')

'''
def extract_json_from_ollama(raw_response):
    if not raw_response.strip().startswith("{"):
        raise ValueError("Output does not start with '{'. Unexpected format:\n" + raw_response[:300])

    try:
        result = json.loads(raw_response)

        if not result:
            raise ValueError("Parsed JSON is empty.")

        if "ENTITIES" in result:
            for key in ["PROPER_NOUN", "NOUN_PHRASE"]:
                if key in result["ENTITIES"] and result["ENTITIES"][key] is None:
                    result["ENTITIES"][key] = []

        if "PROPER_NOUN_DESCRIPTION" in result and result["PROPER_NOUN_DESCRIPTION"] is None:
            result["PROPER_NOUN_DESCRIPTION"] = {}

        return result

    except json.JSONDecodeError as e:
        print("\n===== RAW LLM OUTPUT (for debug) =====\n")
        print(raw_response[:1000])
        print("\n======================================\n")
        raise ValueError(f"Invalid JSON format. {e}")
'''

def extract_json_from_ollama(raw_response):
    if not raw_response.strip().startswith("{"):
        raise ValueError("Output does not start with '{'. Unexpected format:\n" + raw_response[:300])

    try:
        result = json.loads(raw_response)

        if not result:
            raise ValueError("Parsed JSON is empty.")

        if "ENTITIES" in result:
            for key in ["PROPER_NOUN", "NOUN_PHRASE"]:
                if key in result["ENTITIES"] and result["ENTITIES"][key] is None:
                    result["ENTITIES"][key] = []

        if "PROPER_NOUN_DESCRIPTION" in result and result["PROPER_NOUN_DESCRIPTION"] is None:
            result["PROPER_NOUN_DESCRIPTION"] = {}

        return result

    except json.JSONDecodeError as e:
        # Debug preview
        print("\n===== RAW LLM OUTPUT (for debug) =====\n")
        print(raw_response[:1000])
        print("\n======================================\n")

        # Attempt to fix stray apostrophes before colon in keys
        repaired = re.sub(r'"([^"\n\r]+?)\'\s*:', r'"\1":', raw_response)

        try:
            result = json.loads(repaired)
            print("Fixed JSON format by removing stray apostrophes in keys.")
            return result
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format. {e}")


def load_prompt_template(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

'''
def run_ollama_inference(prompt, model_name, ollama_url):
    response = requests.post(ollama_url, json={
        "model": model_name,
        "prompt": prompt,
        "stream": False
    })
    if response.status_code != 200:
        raise Exception(f"Request failed: {response.status_code}\n{response.text}")
    return response.json().get('response', '')
'''

def run_ollama_inference(prompt, model_name, ollama_url, timeout=120):
    try:
        response = requests.post(
            ollama_url,
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False
            },
            timeout=timeout
        )
        response.raise_for_status()  # Raises HTTPError for bad responses

        # Return the response, or fallback to empty string if missing
        return response.json().get('response', '')

    except requests.exceptions.Timeout:
        raise Exception("Ollama inference timed out.")

    except requests.exceptions.HTTPError as http_err:
        raise Exception(f"HTTP error occurred: {http_err} - {response.text}")

    except requests.exceptions.RequestException as req_err:
        raise Exception(f"Request failed: {req_err}")

    except ValueError:
        raise Exception("Invalid JSON response received from Ollama.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-dir", required=True, help="Directory of chunk text files")
    parser.add_argument("--prompt-file", required=True, help="NER prompt file")
    parser.add_argument("--output-dir", required=True, help="Directory to save NER results")
    parser.add_argument("--log-file", required=True, help="Path to shared log.txt file")
    parser.add_argument("--model-name", required=True, help="LLM model name")
    parser.add_argument("--max-retries", type=int, default=2, help="Number of retries for invalid JSON output")
    args = parser.parse_args()

    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    ollama_url = OLLAMA_URL_TEMPLATE.format(host=host)

    os.makedirs(args.output_dir, exist_ok=True)
    prompt_template = load_prompt_template(args.prompt_file)

    chunk_files = sorted([f for f in os.listdir(args.chunks_dir) if f.endswith(".txt")])
    log(f"Found {len(chunk_files)} chunk files.", log_file=args.log_file)

    for fname in tqdm(chunk_files, desc="Processing chunks"):
        chunk_path = os.path.join(args.chunks_dir, fname)
        with open(chunk_path, 'r', encoding='utf-8') as f:
            chunk_text = f.read()

        base_prompt = f"{prompt_template.strip()}\n\n{chunk_text.strip()}"
        
        #print("Base Prompt: ", base_prompt)
        retry_count = 0
        success = False

        while retry_count <= args.max_retries:
            full_prompt = base_prompt
            

            '''
            if retry_count > 0:
                full_prompt = (
    "You are a LOCATION entity extraction API. Return ONLY a valid JSON object.\n"
    "Do not explain, interpret, or comment — not even inside lists or descriptions.\n"
    "All list values must be quoted strings (e.g., [\"X\", \"Y\"]).\n"
    "Descriptions must be short and factual, based strictly on the input text.\n\n"
    "Your output MUST strictly follow this format:\n\n"
    "{\n"
    "  \"ENTITIES\": {\n"
    "    \"PROPER_NOUN\": [\"Location A\", \"Location B\"],\n"
    "    \"NOUN_PHRASE\": [\"the checkpoint\", \"the road\"]\n"
    "  },\n"
    "  \"PROPER_NOUN_DESCRIPTION\": {\n"
    "    \"Location A\": \"A location mentioned in the legal case\",\n"
    "    \"Location B\": \"Another place referenced in the context\"\n"
    "  }\n"
    "}\n\n"
    "IMPORTANT: If no entities are found, return an empty list or object.\n"
    "Begin your JSON response below:\n\n"
    + base_prompt
)
            '''
            
            fallback_1 = """
You must return valid JSON only. Do not add comments or explanations.
All items in any list MUST be quoted strings.
DO NOT write phrases like "is not mentioned" or "not present".
If an entity is not found, return an empty list or empty object.

Return JSON in exactly this structure:

{
  "ENTITIES": {
    "PROPER_NOUN": ["Example Name"],
    "NOUN_PHRASE": ["Example Phrase"]
  },
  "PROPER_NOUN_DESCRIPTION": {
    "Example Name": "Short factual description from text"
  }
}

Begin JSON:
"""
            fallback_2 = """
            Output only JSON. No natural language allowed anywhere.
Lists must contain only quoted strings. 
Do not justify missing entities. If nothing is found, output [] or {}.

Required JSON format (use these keys exactly):

{
  "ENTITIES": {
    "PROPER_NOUN": ["X"],
    "NOUN_PHRASE": ["Y"]
  },
  "PROPER_NOUN_DESCRIPTION": {
    "X": "Short factual description"
  }
}

Start your JSON response below this line:

            """
    
            fallback_3 = """Return a JSON object. Nothing except JSON is allowed. 
List entries must be plain quoted strings. 
No explanations, reasons, or comments allowed. 
Entities not present should be represented as [] or {}.

Model output schema:

{
  "ENTITIES": {
    "PROPER_NOUN": ["Entity1", "Entity2"],
    "NOUN_PHRASE": ["Phrase1", "Phrase2"]
  },
  "PROPER_NOUN_DESCRIPTION": {
    "Entity1": "Description",
    "Entity2": "Description"
  }
}

Produce JSON now: """
            
            fallback_4 = """ Contract: You will output valid JSON only. 
Violation conditions: Any commentary, explanations, or unquoted text inside lists.
All list items must be quoted strings. No exceptions.
Missing entities → return empty lists or an empty object.

Schema to follow:

{
  "ENTITIES": {
    "PROPER_NOUN": ["A"],
    "NOUN_PHRASE": ["B"]
  },
  "PROPER_NOUN_DESCRIPTION": {
    "A": "Short description"
  }
}

Insert your JSON here: """
            
            fallback_variants = [
                    fallback_1,
                    fallback_2,
                    fallback_3,
                    fallback_4
                    ]
            
            if retry_count > 0:
                variant = fallback_variants[(retry_count - 1) % len(fallback_variants)]
                full_prompt = variant + "\n\n" + base_prompt
            
            log(f"Running NER on {fname} (attempt {retry_count + 1})...", log_file=args.log_file)
            
            '''
            try:
                result_raw = run_ollama_inference(full_prompt, args.model_name, ollama_url)
                result = extract_json_from_ollama(result_raw)
                success = True
                break
            except Exception as e:
                log(f"Format issue on attempt {retry_count + 1}: {e}", log_file=args.log_file)
                retry_count += 1
            '''
            try:
                log(f"Prompt length (chars): {len(full_prompt)}", log_file=args.log_file)
                #result_raw = run_ollama_inference(full_prompt, args.model_name, ollama_url)
                result_raw = run_ollama_inference(full_prompt, args.model_name, ollama_url, timeout=300)
                result = extract_json_from_ollama(result_raw)
                success = True
                break
            except Exception as e:
                log(f"Format issue on attempt {retry_count + 1}: {e}", log_file=args.log_file)

                 # Show preview if available
                preview = locals().get("result_raw", "")
                if preview:
                    preview_clean = preview[:300].replace("\n", " ").replace("\r", " ")
                    log(f"[Invalid JSON Preview on attempt {retry_count + 1}] {preview_clean}", log_file=args.log_file)
                retry_count += 1
                # Raise if final attempt fails
                if retry_count > args.max_retries:
                    raise RuntimeError(f"Failed after {args.max_retries} attempts for chunk {fname}: {e}")

        raw_out_path = os.path.join(args.output_dir, fname.replace(".txt", "_raw.txt"))
        with open(raw_out_path, 'w', encoding='utf-8') as raw_file:
            raw_file.write(result_raw)

        out_path = os.path.join(args.output_dir, fname.replace(".txt", ".json"))
        with open(out_path, 'w', encoding='utf-8') as out_file:
            json.dump(result, out_file, indent=2)

        log(f"Saved: {out_path}", log_file=args.log_file)


if __name__ == "__main__":
    main()

