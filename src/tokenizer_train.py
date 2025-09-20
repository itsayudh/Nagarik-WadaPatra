import sentencepiece as spm
import os

def train_tokenizer(corpus_path, model_prefix, vocab_size, model_type):
    """
    Trains a SentencePiece tokenizer on a given corpus.

    Args:
        corpus_path (str): Path to the text corpus file.
        model_prefix (str): Prefix for the output model and vocabulary files.
        vocab_size (int): The desired vocabulary size.
        model_type (str): The tokenizer model type ('unigram' or 'bpe').
    """
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(model_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check if the corpus file exists and is not empty
    if not os.path.exists(corpus_path):
        print(f"Error: Corpus file not found at {corpus_path}.")
        return

    if os.path.getsize(corpus_path) == 0:
        print("Error: The corpus file is empty. Please check your data cleaning script.")
        return

    print(f"Training a {model_type} tokenizer with a vocabulary size of {vocab_size}...")
    spm.SentencePieceTrainer.train(
        f'--input={corpus_path} '
        f'--model_prefix={model_prefix} '
        f'--vocab_size={vocab_size} '
        f'--model_type={model_type} '
        '--character_coverage=0.9995 '
        '--hard_vocab_limit=false '
        '--split_by_unicode_script=true'
    )
    print("Tokenizer training complete.")
    print(f"Model saved to {model_prefix}.model and {model_prefix}.vocab")

if __name__ == "__main__":
    # Define your file paths based on the planned structure
    CORPUS_PATH = 'data/processed/corpus_cleaned.txt'
    MODEL_PREFIX = 'tokenizer/sp'
    VOCAB_SIZE = 32000
    MODEL_TYPE = 'unigram'

    # Run the training process
    train_tokenizer(CORPUS_PATH, MODEL_PREFIX, VOCAB_SIZE, MODEL_TYPE)
