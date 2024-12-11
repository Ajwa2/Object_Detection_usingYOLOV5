from docx import Document

def extract_text_from_docx(docx_path, max_length=30):
    doc = Document(docx_path)
    lines = []
    for para in doc.paragraphs: 
        if para.text.strip(): 
            text = para.text.strip() 
            # Split text into chunks of max_length characters 
            chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)] 
            lines.extend(chunks)
    return lines

docx_path = 'geezBook2.docx'
geez_text_lines = extract_text_from_docx(docx_path)

# Save text lines for further use
with open('geez_text_lines_3.txt', 'w', encoding='utf-8') as f:
    for line in geez_text_lines:
        f.write(line + "\n")
    print("Text lines saved to geez_text_lines_3.txt")
