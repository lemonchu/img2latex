import re

def normalize_latex(latex_str):
    """Normalize LaTeX string to compare visual representation."""
    # Remove all whitespace
    latex_str = re.sub(r'\s+', '', latex_str)
    
    # Standardize equivalent visual notations
    replacements = [
        # Fraction types
        (r'\\dfrac', r'\\frac'),
        (r'\\tfrac', r'\\frac'),
        
        # Text styling
        (r'\\text{([^}]*)}', r'\\mathrm{\\1}'),
        (r'\\textrm{([^}]*)}', r'\\mathrm{\\1}'),
        (r'\\textit{([^}]*)}', r'\\mathit{\\1}'),
        (r'\\textbf{([^}]*)}', r'\\mathbf{\\1}'),
        
        # Parentheses variants
        (r'\\left\(', r'('),
        (r'\\right\)', r')'),
        (r'\\left\[', r'['),
        (r'\\right\]', r']'),
        (r'\\left\\{', r'\\{'),
        (r'\\right\\}', r'\\}'),
        
        # Operators
        #(r'\\cdot', r'*'),
        (r'\\times', r'×'),
        
        # Comparison operators
        (r'\\lt', r'<'),
        (r'\\gt', r'>'),
        (r'\\le', r'\\leq'),
        (r'\\ge', r'\\geq'),
        
        # Fix unnecessary symbols
        (r'{+', r'{'),  # Remove unnecessary + after opening brace
        
        # Normalize spaces in subscripts/superscripts
        (r'_\s*{', r'_{'),
        (r'^\s*{', r'^{'),
    ]
    
    for old, new in replacements:
        latex_str = re.sub(old, new, latex_str)
    
    return latex_str

def are_latex_visually_identical(latex1, latex2):
    """
    Check if two LaTeX expressions would render identically.
    
    Args:
        latex1: First LaTeX formula string
        latex2: Second LaTeX formula string
        
    Returns:
        bool: True if the formulas would look the same when rendered
    """
    norm1 = normalize_latex(latex1)
    norm2 = normalize_latex(latex2)
    
    return norm1 == norm2

def judge_latex_visual(predicted_latex, ground_truth_latex):
    """
    Judge whether a predicted LaTeX formula would look the same as the ground truth.
    
    Args:
        predicted_latex: The model's predicted LaTeX
        ground_truth_latex: The ground truth LaTeX
        
    Returns:
        dict: Results containing visual equality status and normalized forms
    """
    # Normalize both strings
    norm_pred = normalize_latex(predicted_latex)
    norm_gt = normalize_latex(ground_truth_latex)
    
    return {
        "visually_identical": norm_pred == norm_gt,
        "normalized_pred": norm_pred,
        "normalized_gt": norm_gt
    }

# Example usage
if __name__ == "__main__":
    examples = [
        ("\\frac{1}{2}", "\\dfrac{1}{2}"),
        ("x^2", "x^{2}"),
        ("\\text{log}", "\\mathrm{log}"),
        ("\\left( a + b \\right)", "(a+b)"),
        ("a \\cdot b", "a * b"),
        ("\\frac{1}{3x} ", "\\frac{1}{x3}"),
    ]
    
    for ex1, ex2 in examples:
        result = judge_latex_visual(ex1, ex2)
        print(f"Formula 1: {ex1}")
        print(f"Formula 2: {ex2}")
        print(f"Visually identical: {result['visually_identical']}")
        print("---")