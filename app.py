from flask import Flask, render_template, request, redirect, url_for, flash
import os
import spacy
import pickle
from utils.pdf_processing import extract_text_from_pdf
from utils.data_preprocessing import preprocess_text
from utils.ner_pipeline import load_ner_model
from utils.clause_extraction import extract_clauses
from utils.risk_analysis import analyze_risks

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data/raw_documents'
app.secret_key = 'supersecretkey'

# Load pre-trained models
with open('models/text_classifier.pkl', 'rb') as f:
    text_classifier = pickle.load(f)

nlp = spacy.load('en_core_web_sm')  # Load spaCy model for standard NER
custom_ner = load_ner_model()       # Load fine-tuned NER model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_document():
    if 'document' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))

    file = request.files['document']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))

    if file and file.filename.endswith('.pdf'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        return redirect(url_for('analyze_document', filename=file.filename))
    else:
        flash('Unsupported file format. Please upload a PDF file.')
        return redirect(url_for('index'))

@app.route('/analyze/<filename>')
def analyze_document(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        # Extract and preprocess text
        raw_text = extract_text_from_pdf(filepath)
        processed_text = preprocess_text(raw_text)

        # Perform NER and classification
        doc = custom_ner(processed_text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        category = text_classifier.predict([processed_text])[0]

        # Extract important clauses and analyze risks
        clauses = extract_clauses(processed_text)
        risks = analyze_risks(clauses)

        # Log the analysis
        app.logger.info(f'Analyzed document: {filename}')

        return render_template('result.html', entities=entities, category=category, clauses=clauses, risks=risks)

    except Exception as e:
        app.logger.error(f'Error analyzing document {filename}: {e}')
        flash('An error occurred during analysis. Please try again.')
        return redirect(url_for('index'))

if __name__ == '__main__':
    # Configure logging
    import logging
    logging.basicConfig(filename='logs/app.log', level=logging.INFO)
    app.run(debug=True)