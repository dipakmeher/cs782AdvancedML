from transformers import AutoTokenizer
import argparse

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Path to input .txt file")
    args = parser.parse_args()

    # Load tokenizer (same as LLaMA 3 in Ollama)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    # Load input text
    input_text = load_text(args.input)

    # Encode and count tokens
    token_ids = tokenizer.encode(input_text, return_tensors='pt')
    token_count = token_ids.shape[1]
    tokens = tokenizer.tokenize(input_text)

    print(f"Total tokens: {token_count}")
    print("First 50 tokens:", tokens[:50])

if __name__ == '__main__':
    main()

