from transformers import pipeline
import PyPDF2
import re

# Load a pre-trained Question Answering model from Hugging Face
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Function to extract all dates from the text
def extract_dates(text):
    date_pattern = r'\b(?:\d{1,2}[-/thstndrd]*\s?[a-zA-Z]+[-/ ]*\d{2,4}|\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4})\b'
    dates = re.findall(date_pattern, text)
    return dates

# Function to search for the manager's approval date based on keywords
def find_approval_date(text, approval_keywords=["approved", "access granted", "manager approved"]):
    # First, extract all dates
    dates = extract_dates(text)
    
    # Now, look for the approval context
    for keyword in approval_keywords:
        # Search for the keyword and get the surrounding context
        if keyword.lower() in text.lower():
            # Find the position of the keyword and get nearby dates
            keyword_pos = text.lower().find(keyword.lower())
            before_text = text[:keyword_pos]
            after_text = text[keyword_pos:]
            
            # Print for debugging
            print("Context around approval:", before_text[-500:], after_text[:500])  # Show surrounding text
            
            # Extract dates surrounding this context
            approval_dates = extract_dates(before_text + after_text)
            if approval_dates:
                return approval_dates[-1]  # Returning the most recent date found near the approval keyword
    
    return None

# Function to extract text from a PDF file
def extract_pdf_text(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text

# Function for Question Answering
def qa_inference(context, question):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# Example usage
pdf_path = "sample3.pdf"  # Path to your PDF file
extracted_text = extract_pdf_text(pdf_path)  # Assuming you have a function for text extraction


# Define questions
question_1 = "Should the employee's access be retained or removed?"
question_2 = "For which employee are we considering in this mail chain ?"
#approval_date = find_approval_date(extracted_text)

# Get answers from the QA pipeline
answer_1 = qa_inference(extracted_text, question_1)
answer_2 = qa_inference(extracted_text, question_2)
#answer_3 = qa_inference("Approval_date:", approval_date)

print("Answer to Question 1:", answer_1)
print('Answer to Question 2:', answer_2)
#print("Approval_date:", approval_date)

