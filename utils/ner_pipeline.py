import spacy

def load_ner_model():
    """Loads or fine-tunes a custom NER model."""
    nlp = spacy.blank('en')  # Start with a blank model

    # Add a new NER pipeline if it doesn't exist
    if 'ner' not in nlp.pipe_names:
        ner = nlp.add_pipe('ner')
    else:
        ner = nlp.get_pipe('ner')

    # Add custom labels for legal entities
    labels = ['CLAUSE', 'OBLIGATION', 'PARTY', 'DATE', 'RISK']
    for label in labels:
        ner.add_label(label)

    # Optionally load a pre-trained model
    # nlp.from_disk('models/custom_ner_model')
    return nlp