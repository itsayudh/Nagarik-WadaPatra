import re
import os

def clean_corpus(input_filepath, output_filepath):
    """
    Cleans a text corpus by removing page headers, multiple newlines,
    and excess whitespace.

    Args:
        input_filepath (str): The path to the raw corpus file.
        output_filepath (str): The path to save the cleaned corpus file.
    """
    print(f"Reading raw corpus from {input_filepath}...")
    try:
        with open(input_filepath, 'r', encoding='utf-8') as infile:
            text = infile.read()
    except FileNotFoundError:
        print(f"Error: The file {input_filepath} was not found.")
        return

    # 1. Remove page headers (e.g., "--- PAGE X ---")
    # This regex looks for lines that start and end with "---" and contain "PAGE".
    cleaned_text = re.sub(r'--- PAGE \d+ ---\n?', '', text)

    # 2. Remove extra whitespace and standardize newlines.
    # Replace multiple consecutive newlines with a single newline.
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
    # Replace any three or more newlines with just two to maintain paragraph breaks.
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)

    # 3. Clean up leading and trailing whitespace on each line.
    lines = [line.strip() for line in cleaned_text.split('\n')]
    cleaned_text = '\n'.join(lines)

    # Save the cleaned text to a new file
    print(f"Writing cleaned corpus to {output_filepath}...")
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(cleaned_text)

    print("Corpus cleaning complete.")

if __name__ == "__main__":
    # Define your file paths based on the planned structure
    input_file = 'data/processed/corpus.txt'
    output_file = 'data/processed/corpus_cleaned.txt'
    
    # Run the cleaning process
    clean_corpus(input_file, output_file)
