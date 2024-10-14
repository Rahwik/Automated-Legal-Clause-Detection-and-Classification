import re

def preprocess_text(text):
    """Cleans and preprocesses text data."""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'\d+', '', text)   # Remove numbers
    text = text.lower()               # Convert to lowercase
    return text.strip()