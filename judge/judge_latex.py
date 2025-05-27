import sympy
from sympy.parsing.latex import parse_latex
import re

def normalize_latex(latex_str):
    """Normalize LaTeX string by removing unnecessary spaces and standardizing notation."""
    # Remove all whitespace
    latex_str = re.sub(r'\s+', '', latex_str)
    
    # Standardize some common equivalent notations
    replacements = [
        (r'\\dfrac', r'\\frac'),
        (r'\\text{([^}]*)}', r'\\mathrm{\\1}'),
        (r'\\left\(', r'('),
        (r'\\right\)', r')'),
        (r'\\left\[', r'['),
        (r'\\right\]', r']'),
        (r'\\cdot', r'*'),
        (r'{+', r'{'),  # Remove unnecessary + after opening brace
        (r'\\lt', r'<'),
        (r'\\gt', r'>'),
        (r'\\le', r'\\leq'),
        (r'\\ge', r'\\geq'),
    ]
    
    for old, new in replacements:
        latex_str = re.sub(old, new, latex_str)
    
    return latex_str

def are_latex_equal(latex1, latex2, strict=False):
    """
    Compare two LaTeX formulas to determine if they represent the same mathematical expression.
    
    Args:
        latex1: First LaTeX formula string
        latex2: Second LaTeX formula string
        strict: If True, use only symbolic comparison; if False, fall back to string normalization
        
    Returns:
        bool: True if the formulas are equivalent, False otherwise
    """
    # First attempt: Try symbolic comparison using sympy
    try:
        expr1 = parse_latex(latex1)
        expr2 = parse_latex(latex2)
        
        # Check if the difference simplifies to zero
        diff = sympy.simplify(expr1 - expr2)
        if diff == 0:
            return True
        
        # Check if the ratio is 1 (for expressions that can be divided)
        try:
            ratio = sympy.simplify(expr1 / expr2)
            if ratio == 1:
                return True
        except:
            pass
            
        # If strict mode is enabled, return False here
        if strict:
            return False
            
    except Exception as e:
        if strict:
            raise ValueError(f"Could not parse LaTeX expressions: {e}")
        # If parsing fails, fall back to string comparison
    
    # Fallback: Normalize strings and compare
    norm1 = normalize_latex(latex1)
    norm2 = normalize_latex(latex2)
    
    return norm1 == norm2

def judge_latex(predicted_latex, ground_truth_latex):
    """
    Judge whether a predicted LaTeX formula matches the ground truth.
    
    Args:
        predicted_latex: The model's predicted LaTeX
        ground_truth_latex: The ground truth LaTeX
        
    Returns:
        dict: Results containing equality status and normalized forms
    """
    try:
        # Try symbolic comparison first
        symbolic_equal = are_latex_equal(predicted_latex, ground_truth_latex, strict=True)
        if symbolic_equal:
            return {
                "equal": True,
                "method": "symbolic",
                "normalized_pred": predicted_latex,
                "normalized_gt": ground_truth_latex
            }
    except:
        # If symbolic comparison fails, continue to string normalization
        pass
    
    # Normalize both strings
    norm_pred = normalize_latex(predicted_latex)
    norm_gt = normalize_latex(ground_truth_latex)
    
    return {
        "equal": norm_pred == norm_gt,
        "method": "string_normalization",
        "normalized_pred": norm_pred,
        "normalized_gt": norm_gt
    }

# Example usage
if __name__ == "__main__":
    examples = [
        ("\\frac{1}{2}", "\\dfrac{1}{2}"),
        ("x^2 + 2x + 1", "(x+1)^2"),
        ("\\sin^2(x) + \\cos^2(x)", "1"),
        ("\\frac{a}{b} + \\frac{c}{d}", "\\frac{ad + bc}{bd}")
    ]
    
    for ex1, ex2 in examples:
        result = judge_latex(ex1, ex2)
        print(f"Formula 1: {ex1}")
        print(f"Formula 2: {ex2}")
        print(f"Equal: {result['equal']}")
        print(f"Method: {result['method']}")
        print("---")