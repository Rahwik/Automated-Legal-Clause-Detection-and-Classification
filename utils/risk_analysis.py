def analyze_risks(clauses):
    """Analyzes clauses for potential risks."""
    risk_keywords = ['penalty', 'breach', 'liability', 'fine']
    risks = [clause for clause in clauses if any(keyword in clause.lower() for keyword in risk_keywords)]
    return risks