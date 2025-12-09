from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import argparse
import torch
import os

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def save_output(output_text, input_filename):
    # Get the base filename (no directory path)
    base_name = os.path.basename(input_filename)
    output_filename = f"output_{base_name}"
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(output_text)
    
    print(f"Output saved to {output_filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=True, help="Path to prompt template file containing '{input_text}'")
    parser.add_argument('--input', type=str, required=True, help="Path to input text file")
    args = parser.parse_args()

    # Load prompt and input text
    prompt_template = load_text(args.prompt)
    input_text = load_text(args.input)

    # Replace the placeholder with actual input
    full_prompt = prompt_template.replace("{input_text}", input_text)

    # Load tokenizer and model
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    # Use pipeline for easy inference
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Generate output
    result = pipe(full_prompt, max_new_tokens=300, do_sample=True, top_p=0.9, temperature=0.7)[0]['generated_text']

    # Save output
    save_output(result, args.input)

if __name__ == '__main__':
    main()

