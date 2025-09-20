import PyPDF2
import os

def extract_text_from_pdfs(input_dir, output_file):
    """
    Extracts text from all PDF files in a given directory and
    saves it to a single output file.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Open the output file in write mode
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Loop through all files in the input directory
        for filename in os.listdir(input_dir):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(input_dir, filename)
                print(f"Extracting text from {filename}...")
                
                try:
                    with open(pdf_path, 'rb') as pdf_file:
                        reader = PyPDF2.PdfReader(pdf_file)
                        for page_num, page in enumerate(reader.pages):
                            text = page.extract_text()
                            if text:
                                # Simple cleaning: replace multiple newlines with one
                                cleaned_text = "\n".join(line.strip() for line in text.split('\n') if line.strip())
                                outfile.write(f"\n--- PAGE {page_num + 1} ---\n")
                                outfile.write(cleaned_text)
                                outfile.write("\n\n")
                except Exception as e:
                    print(f"Error extracting text from {filename}: {e}")
    
    print(f"Text extraction complete. All text saved to {output_file}")
    
if __name__ == "__main__":
    # Define your file paths based on the planned structure
    input_directory = 'data/raw'
    output_filepath = 'data/processed/corpus.txt'
    
    # Run the extraction process
    extract_text_from_pdfs(input_directory, output_filepath)
