from flask import Flask, request, jsonify, render_template
import PyPDF2
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)


def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
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

# Step 3: Tokenize text
def tokenize_text(text):
    tokens = word_tokenize(text)
    return set(tokens)

# Step 4: Calculate the similarity score using TF-IDF and cosine similarity
def calculate_similarity(resume_text, job_description_text):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description_text])
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return similarity_score * 100  # Convert to percentage

# Step 5: Generate suggestions for improvement
def generate_suggestions(resume_tokens, job_description_tokens):
    missing_keywords = job_description_tokens - resume_tokens
    suggestions = []
    
    if missing_keywords:
        suggestions.append("Consider including the following keywords: " + ", ".join(missing_keywords))
    
    # Example suggestions based on missing sections or keywords
    if 'skills' not in resume_tokens:
        suggestions.append("Consider adding a 'Skills' section to highlight your relevant abilities.")
    
    if 'experience' not in resume_tokens:
        suggestions.append("Consider adding an 'Experience' section to detail your past job roles and responsibilities.")
    
    if not suggestions:
        suggestions.append("Your resume is well-aligned with the job description!")
    
    return suggestions

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

# Serve the front-end web page
@app.route('/')
def index():
    return render_template('index.html')

# API endpoint to score resume against job description and provide suggestions
@app.route('/score', methods=['POST'])
def score_resume():
    if 'resume' not in request.files or 'job_description' not in request.files:
        return jsonify({'error': 'Please provide both resume and job description files'}), 400

    resume_file = request.files['resume']
    job_description_file = request.files['job_description']

    # Extract text from files
    resume_text = extract_text_from_pdf(resume_file)
    job_description_text = job_description_file.read().decode('utf-8')

    # Preprocess texts
    resume_text_preprocessed = preprocess_text(resume_text)
    job_description_text_preprocessed = preprocess_text(job_description_text)

    # Calculate similarity score
    score = calculate_similarity(resume_text_preprocessed, job_description_text_preprocessed)

    # Tokenize texts for suggestions
    resume_tokens = tokenize_text(resume_text_preprocessed)
    job_description_tokens = tokenize_text(job_description_text_preprocessed)

    # Generate suggestions for improvement
    suggestions = generate_suggestions(resume_tokens, job_description_tokens)

    return jsonify({
        'match_score': f"{score:.2f}%",
        'suggestions': suggestions
    })
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)