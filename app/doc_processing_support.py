from pypdf import PdfReader
import docx2txt

def clean_utf8(input_str):
    """
    function to clean strings to UTF-8 compatible encoding.
    """
    input_str = bytes(input_str, "utf-8").decode("utf-8", "ignore")
    return input_str

def process_pdf(pdf_dir, pages = True)
    reader = PdfReader(pdf_dir)
    pagetexts = []
    total_splits = []
    for page in reader.pages:
        text = page.extract_text()
        text = text.replace("\n", " ")  # clean text of new lines
        text = clean_utf8(text)
        pagetexts.append(text)
    if pages = True:
        return pagetexts
    else:
        return ' '.join(pagetexts)

def process_docx(docx_dir):
    text = docx2txt.process(docx_dir)
    text = clean_utf8(text)
    return text

def process_txt(txt):
    text = txt.read().decode("utf-8").replace("\n", " ")
    text = clean_utf8(text)
    return text

def return_texts(filelist):
    '''
    returns list of txt from inputs in correct groq-friendly format
    '''
    textfiles = []
    
    for file in filelist:
        ext = file.name.split(".")[1]
        if "txt" in ext:
            text = process_txt(file)
        elif "pdf" in ext:
            text = process_pdf(file, pages = False)
        elif "doc" in ext:
            text = process_docx(file)
        textfiles.append(
            {"source": {
                "type": "text",
                "text": text,
                }
            }
        )
    return textfiles









        