# translate_pdf.py
from deep_translator import GoogleTranslator
import fitz  # PyMuPDF
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import textwrap
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def translate_pdf(input_pdf, output_pdf, chunk_size=1000, max_retries=3, timeout=30):
    """
    Translate PDF from Dutch to English with timeout and retry handling.
    
    Args:
        input_pdf: Path to input PDF file
        output_pdf: Path to output translated PDF file
        chunk_size: Size of text chunks to translate (smaller = faster, default 1000)
        max_retries: Maximum number of retry attempts (default 3)
        timeout: Request timeout in seconds (default 30)
    """
    # Open the PDF
    doc = fitz.open(input_pdf)
    
    # Configure session with retry strategy
    session = requests.Session()
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    print("Extracting text from PDF...")
    all_text = ""
    for page in doc:
        text = page.get_text("text")
        all_text += text + "\n"
    
    if not all_text.strip():
        print("⚠️ No text found in PDF")
        return
    
    # Break text into smaller chunks for faster translation
    chunks = []
    current_chunk = ""
    for line in all_text.split("\n"):
        if len(current_chunk) + len(line) + 1 > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = line
        else:
            current_chunk += "\n" + line if current_chunk else line
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    print(f"Translating {len(chunks)} chunks...")
    
    # Create translator with timeout configuration
    translator = GoogleTranslator(source='nl', target='en')
    
    translated_text = ""
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
            
        print(f"Translating chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
        
        # Retry logic with exponential backoff
        for attempt in range(max_retries):
            try:
                # Add small delay to avoid rate limiting
                if i > 0:
                    time.sleep(0.5)  # 500ms delay between chunks
                
                translated_chunk = translator.translate(chunk)
                translated_text += translated_chunk + "\n"
                break  # Success, exit retry loop
                
            except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                wait_time = (2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                error_type = type(e).__name__
                print(f"⚠️ {error_type} on chunk {i+1}, attempt {attempt+1}/{max_retries}. Retrying in {wait_time}s...")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    print(f"❌ Failed to translate chunk {i+1} after {max_retries} attempts due to {error_type}. Skipping...")
                    translated_text += f"[Translation failed: {error_type}]\n"
                    
            except Exception as e:
                error_msg = str(e)
                # Check if it's a timeout-related error even if not directly a Timeout exception
                if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                    wait_time = (2 ** attempt)
                    print(f"⚠️ Timeout error on chunk {i+1}, attempt {attempt+1}/{max_retries}: {error_msg[:100]}. Retrying in {wait_time}s...")
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                    else:
                        print(f"❌ Failed to translate chunk {i+1} after {max_retries} attempts. Skipping...")
                        translated_text += f"[Translation failed: timeout]\n"
                else:
                    wait_time = (2 ** attempt)
                    print(f"⚠️ Error on chunk {i+1}, attempt {attempt+1}/{max_retries}: {error_msg[:100]}. Retrying in {wait_time}s...")
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                    else:
                        print(f"❌ Failed to translate chunk {i+1} after {max_retries} attempts: {error_msg[:100]}. Skipping...")
                        translated_text += f"[Translation failed: {error_msg[:50]}]\n"
    
    print("Saving translated PDF...")
    c = canvas.Canvas(output_pdf, pagesize=A4)
    width, height = A4
    y = height - 50
    for line in textwrap.wrap(translated_text, 100):
        if y < 50:
            c.showPage()
            y = height - 50
        c.drawString(50, y, line)
        y -= 15
    c.save()
    
    print(f"✅ Translation completed. Saved as '{output_pdf}'")

# Example usage:
if __name__ == "__main__":
    translate_pdf("hezop1.pdf", "hezop1_translated.pdf")
