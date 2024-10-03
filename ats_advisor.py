import PyPDF2
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
from nltk.corpus import stopwords

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

# Step 2: Clean and preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\W+', ' ', text)  # Remove non-alphanumeric characters
    text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Step 3: Extract text from the job description text file
def extract_text_from_txt(txt_path):
    with open(txt_path, 'r') as file:
        return file.read()

# Step 4: Calculate the similarity score using TF-IDF and cosine similarity
def calculate_similarity(resume_text, job_description_text):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description_text])
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return similarity_score * 100  # Convert to percentage

# Main function to run the service
def score_resume_against_job_description(pdf_path, txt_path):
    resume_text = extract_text_from_pdf(pdf_path)
    job_description_text = extract_text_from_txt(txt_path)

    # Preprocess texts
    resume_text = preprocess_text(resume_text)
    job_description_text = preprocess_text(job_description_text)

    # Calculate and return the similarity score
    score = calculate_similarity(resume_text, job_description_text)
    return f"Match Score: {score:.2f}%"

# Example usage
pdf_resume_path = 'resume.pdf'  # Path to the PDF resume file
job_description_path = 'job_desc.txt'  # Path to the job description text file

print(score_resume_against_job_description(pdf_resume_path, job_description_path))
