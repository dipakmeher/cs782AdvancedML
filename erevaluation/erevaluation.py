import argparse
import pandas as pd
import requests
from datetime import datetime
import os

# ------------------------
# OLLAMA CALL FUNCTION
# ------------------------
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
        response.raise_for_status()

        return response.json().get('response', '')

    except requests.exceptions.Timeout:
        raise Exception("Ollama inference timed out.")
    except requests.exceptions.HTTPError as http_err:
        raise Exception(f"HTTP error occurred: {http_err} - {response.text}")
    except requests.exceptions.RequestException as req_err:
        raise Exception(f"Request failed: {req_err}")
    except ValueError:
        raise Exception("Invalid JSON response received from Ollama.")


# ------------------------
# MAIN PIPELINE
# ------------------------
def main():

    parser = argparse.ArgumentParser(description="Run LLM extraction on CSV rows.")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to input CSV file.")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Path to save CSV with LLM outputs.")
    parser.add_argument("--prompt_path", type=str, required=True,
                        help="Path to the full prompt text file.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Ollama model name.")
    parser.add_argument("--ollama_host", type=str, required=True,
                        help="Ollama host, e.g. 127.0.0.1:11434")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout for Ollama call (default 300s).")
    parser.add_argument("--limit", type=int, default=None,
                    help="Limit number of rows to process (optional)")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Results dir")

    args = parser.parse_args()

    # Build full Ollama URL
    ollama_url = f"http://{args.ollama_host}/api/generate"

    # Load prompt template
    with open(args.prompt_path, "r", encoding="utf-8") as f:
        PROMPT_TEMPLATE = f.read()

    # Load CSV
    df = pd.read_csv(args.csv_path)
    #df = pd.read_csv(args.csv_path).sample(frac=1, random_state=42).head(500)

    if args.limit:
        df = df.head(args.limit)
        #df = pd.read_csv(args.csv_path).sample(frac=1, random_state=42).head(500)

    outputs = []

    for idx, row in df.iterrows():
        entity_types = row["Entity_Types"]
        input_text = row["Input_Text"]

        # Prepare full prompt
        full_prompt = PROMPT_TEMPLATE
        full_prompt = full_prompt.replace("{entity_types}", entity_types)
        full_prompt = full_prompt.replace("{input_text}", input_text)

        print(f"\nProcessing row {idx+1}/{len(df)} ...")
        print(f"Entity Types: {entity_types}")

        # Run inference
        try:
            llm_output = run_ollama_inference(
                full_prompt,
                args.model_name,
                ollama_url,
                timeout=args.timeout
            )
        except Exception as e:
            print(f"ERROR on row {idx}: {e}")
            llm_output = f"ERROR: {e}"

        outputs.append(llm_output)

    base_results_dir = args.results_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_results_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    df["LLM_Output"] = outputs
    output_path = os.path.join(run_dir, args.output_csv)
    df.to_csv(output_path, index=False)

    # Save outputs
    #df["LLM_Output"] = outputs
    #df.to_csv(run_dir/args.output_csv, index=False)

    print("\nSaved output CSV to:", args.output_csv)


# ------------------------
# ENTRY POINT
# ------------------------
if __name__ == "__main__":
    main()
