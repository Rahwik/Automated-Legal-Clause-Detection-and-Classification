def extract_clauses(text):
    """Extracts important clauses from the legal document."""
    clauses = []
    # Use simple rules or advanced NLP techniques to identify clauses
    for line in text.split('.'):
        if 'clause' in line.lower() or 'agreement' in line.lower():
            clauses.append(line.strip())
    return clauses