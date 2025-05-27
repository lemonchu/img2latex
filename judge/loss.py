import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from io import BytesIO
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim
import tempfile
import os
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def render_latex_matplotlib(latex_str, dpi=200):
    """Render LaTeX formula to image using matplotlib."""
    try:
        # Use non-interactive backend to avoid display issues
        plt.switch_backend('Agg')
        
        # Use MathText instead of LaTeX
        plt.rcParams['text.usetex'] = False  # Don't use system LaTeX
        plt.rcParams['mathtext.fontset'] = 'cm'  # Use Computer Modern font
        
        # Create a figure with transparent background
        fig = plt.figure(figsize=(5, 1), dpi=dpi)
        plt.axis('off')
        plt.text(0.5, 0.5, f"${latex_str}$", size=24, ha='center', va='center')
        
        # Save to a bytes buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, transparent=True)
        plt.close(fig)
        
        # Convert to image
        buf.seek(0)
        img = Image.open(buf)
        gray_img = np.array(img.convert('L'))  # Convert to grayscale
        
        # Enable for debugging
        # cv2.imwrite(f"debug_{latex_str[:10].replace('\\', '_')}.png", gray_img)
        
        return gray_img
    except Exception as e:
        logger.error(f"Matplotlib rendering failed: {e}")
        return None

def render_latex_native(latex_str, density=300):
    """Render LaTeX formula to image using native LaTeX for higher quality."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal LaTeX document
            tex_content = r"""
\documentclass[preview]{standalone}
\usepackage{amsmath,amssymb,amsfonts}
\begin{document}
$%s$
\end{document}
""" % latex_str
            
            tex_file = os.path.join(tmpdir, "formula.tex")
            with open(tex_file, 'w') as f:
                f.write(tex_content)
            
            # Compile LaTeX to PDF
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', '-output-directory', tmpdir, tex_file], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"pdflatex failed: {result.stderr}")
                return None
            
            # Convert PDF to PNG
            pdf_file = os.path.join(tmpdir, "formula.pdf")
            png_file = os.path.join(tmpdir, "formula.png")
            
            convert_result = subprocess.run(
                ['convert', '-density', str(density), pdf_file, '-quality', '100', png_file],
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            if convert_result.returncode != 0:
                logger.error(f"ImageMagick convert failed: {convert_result.stderr}")
                return None
            
            # Read the image
            if os.path.exists(png_file):
                img = cv2.imread(png_file, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    logger.error(f"Failed to read image: {png_file}")
                return img
            else:
                logger.error(f"PNG file not created: {png_file}")
                return None
    except Exception as e:
        logger.error(f"Native LaTeX rendering failed: {e}")
        return None

def compute_image_similarity(img1, img2):
    """Compute similarity metrics between two images."""
    try:
        # Check inputs
        if img1 is None or img2 is None:
            logger.error("One or both images are None")
            return {
                "mse": float('inf'),
                "ssim": 0.0,
                "normalized_distance": 1.0
            }
        
        # Resize to the same dimensions if needed
        if img1.shape != img2.shape:
            # Resize to the larger of the two dimensions
            h = max(img1.shape[0], img2.shape[0])
            w = max(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (w, h), interpolation=cv2.INTER_AREA)
            img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)
        
        # Calculate MSE (Mean Squared Error)
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        
        # Calculate SSIM (Structural Similarity Index)
        ssim_score = ssim(img1, img2)
        
        return {
            "mse": mse,
            "ssim": ssim_score,
            "normalized_distance": mse / 255**2  # Normalize to 0-1 range
        }
    except Exception as e:
        logger.error(f"Image similarity computation failed: {e}")
        return {
            "mse": float('inf'),
            "ssim": 0.0,
            "normalized_distance": 1.0
        }

def latex_image_loss(pred_latex, gt_latex, render_method="matplotlib"):
    """
    Calculate loss between two LaTeX formulas by comparing their rendered images.
    
    Args:
        pred_latex: Predicted LaTeX string
        gt_latex: Ground truth LaTeX string
        render_method: Method to use for rendering ("matplotlib" or "native")
        
    Returns:
        dict: Loss metrics comparing the rendered images
    """
    try:
        # Render both LaTeX strings to images
        if render_method == "native":
            img_pred = render_latex_native(pred_latex)
            img_gt = render_latex_native(gt_latex)
        else:
            img_pred = render_latex_matplotlib(pred_latex)
            img_gt = render_latex_matplotlib(gt_latex)
        
        # Debug images
        if img_pred is None:
            logger.error(f"Failed to render predicted: {pred_latex}")
        if img_gt is None:
            logger.error(f"Failed to render ground truth: {gt_latex}")
        
        # If rendering failed for either, return maximum loss
        if img_pred is None or img_gt is None:
            return {
                "mse": float('inf'),
                "ssim": 0.0,
                "normalized_distance": 1.0,
                "render_success": False
            }
        
        # Save debug images if needed
        # cv2.imwrite(f"debug_pred.png", img_pred)
        # cv2.imwrite(f"debug_gt.png", img_gt)
        
        # Compute similarity metrics
        metrics = compute_image_similarity(img_pred, img_gt)
        metrics["render_success"] = True
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error in latex_image_loss: {e}")
        # If any error occurs, return maximum loss
        return {
            "mse": float('inf'),
            "ssim": 0.0,
            "normalized_distance": 1.0,
            "render_success": False,
            "error": str(e)
        }

# Example usage
if __name__ == "__main__":
    examples = [
        ("\\frac{1}{2}", "\\dfrac{1}{2}"),
        ("x^2 + 2x + 1", "(x+1)^2"),
        ("\\int_{0}^{\\infty} e^{-x^2} dx", "\\int_0^\\infty e^{-x^2} dx"),
        ("\\sum_{i=1}^{n} i", "\\sum_{i=1}^n i"),
        ("114514", "191919810"),
        ("[123]", "\\left[123\\right]"),
    ]
    
    # Check dependencies
    has_native = True
    try:
        subprocess.run(['pdflatex', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(['convert', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except:
        has_native = False
        print("Native LaTeX rendering not available - pdflatex or ImageMagick missing")
    
    for ex1, ex2 in examples:
        print(f"Formula 1: {ex1}")
        print(f"Formula 2: {ex2}")
        
        # Try with matplotlib renderer first
        result_mpl = latex_image_loss(ex1, ex2, "matplotlib")
        
        # Fix the print statement to handle different types
        mse_val = result_mpl.get('mse')
        if mse_val == float('inf'):
            mse_str = "inf"
        elif isinstance(mse_val, (int, float)):
            mse_str = f"{mse_val:.4f}"
        else:
            mse_str = str(mse_val)
            
        ssim_val = result_mpl.get('ssim', 0)
        ssim_str = f"{ssim_val:.4f}" if isinstance(ssim_val, (int, float)) else str(ssim_val)
        
        print(f"Matplotlib renderer: MSE = {mse_str}, SSIM = {ssim_str}")
        
        # Try with native LaTeX if available
        if has_native:
            result_native = latex_image_loss(ex1, ex2, "native")
            
            # Same fix for native renderer
            mse_val = result_native.get('mse')
            if mse_val == float('inf'):
                mse_str = "inf"
            elif isinstance(mse_val, (int, float)):
                mse_str = f"{mse_val:.4f}"
            else:
                mse_str = str(mse_val)
                
            ssim_val = result_native.get('ssim', 0)
            ssim_str = f"{ssim_val:.4f}" if isinstance(ssim_val, (int, float)) else str(ssim_val)
            
            print(f"Native LaTeX renderer: MSE = {mse_str}, SSIM = {ssim_str}")
        
        print("---")