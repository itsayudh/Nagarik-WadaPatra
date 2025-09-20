import sentencepiece as spm
import os

def run_round_trip_test(model_path):
    """
    Performs a round-trip test on the trained SentencePiece tokenizer.

    Args:
        model_path (str): The path to the trained SentencePiece model file.
    """
    if not os.path.exists(model_path):
        print(f"Error: Tokenizer model not found at {model_path}.")
        return

    # Load the trained tokenizer model
    sp = spm.SentencePieceProcessor(model_file=model_path)
    
    # Define test strings with both English and Nepali text, and numerals.
    test_strings = [
        "Welcome to the citizen services portal.",
        "नागरिक वडापत्र अनुसार सेवा लिनुहोस्।",
        "सेवा शुल्क: रु. ५०",
        "मिति: २०८०/०२/२३",
        "Number: 1234567890",
        "Nepali numerals: ०१२३४५६७८९",
    ]

    print("--- Running Round-Trip Test ---")
    for original_text in test_strings:
        print(f"\nOriginal: '{original_text}'")
        
        # 1. Encode the text
        ids = sp.encode_as_ids(original_text)
        
        # 2. Decode the token IDs back to text
        decoded_text = sp.decode_ids(ids)
        
        # 3. Check for a perfect match
        if original_text == decoded_text:
            print("  ✅ Match!")
        else:
            print("  ❌ Mismatch!")
            print(f"  Decoded:  '{decoded_text}'")

    print("\n--- Test Complete ---")


if __name__ == "__main__":
    # Define your tokenizer model path based on the planned structure
    MODEL_PATH = 'tokenizer/sp.model'
    run_round_trip_test(MODEL_PATH)
