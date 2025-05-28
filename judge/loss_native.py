import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import tempfile
import os
import subprocess
import logging
from PIL import Image
import shutil
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def render_latex_native(latex_str, density=300, save_dir=None, display_mode="inline"):
    """Render LaTeX formula to image using native LaTeX for higher quality.
    
    Args:
        latex_str: LaTeX 字符串.
        density: ImageMagick convert 的 density 参数.
        save_dir: 如果不是 None，则将生成的 PNG 图片保存到该目录.
        display_mode: 渲染模式，"inline" 使用 $...$，"display" 使用 \[...\].
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            if display_mode == "display":
                math_env = r"$%s$" % latex_str
                everymath_line = r"\everymath{\displaystyle}"
            else:
                math_env = r"$%s$" % latex_str
                everymath_line = ""
            
            # Create a minimal LaTeX document with optional \everymath{\displaystyle}
            tex_content = r"""
\documentclass[preview]{standalone}
\usepackage{amsmath,amssymb,amsfonts}
%s
\begin{document}
%s
\end{document}
""" % (everymath_line, math_env)
            
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
            
            # Convert PDF to PNG using ImageMagick
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
            
            # 如果提供了保存目录，则将生成的图片复制到指定目录
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                # 使用 LaTeX 字符串的 hash 生成文件名，避免重名
                dest_file = os.path.join(save_dir, f"{hash(latex_str)}_{display_mode}.png")
                shutil.copy(png_file, dest_file)
                logger.info(f"Saved rendered image to {dest_file}")
            
            # Read the image in grayscale for pixel-level comparison
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
        if img1 is None or img2 is None:
            logger.error("One or both images are None")
            return {
                "mse": float('inf'),
                "ssim": 0.0,
                "normalized_distance": 1.0
            }
        if img1.shape != img2.shape:
            h = max(img1.shape[0], img2.shape[0])
            w = max(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (w, h), interpolation=cv2.INTER_AREA)
            img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)
        
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        ssim_score = ssim(img1, img2)
        
        return {
            "mse": mse,
            "ssim": ssim_score,
            "normalized_distance": mse / 255**2
        }
    except Exception as e:
        logger.error(f"Image similarity computation failed: {e}")
        return {
            "mse": float('inf'),
            "ssim": 0.0,
            "normalized_distance": 1.0
        }

def latex_image_loss(pred_latex, gt_latex, save_dir=None, display_mode="inline"):
    """
    Calculate loss between two LaTeX formulas by comparing their rendered images using native LaTeX.
    
    Args:
        pred_latex: Predicted LaTeX string.
        gt_latex: Ground truth LaTeX string.
        save_dir: 如果不是 None，则将渲染图片保存到该目录.
        display_mode: 渲染模式，"inline" 或 "display".
    """
    try:
        img_pred = render_latex_native(pred_latex, save_dir=save_dir, display_mode=display_mode)
        img_gt = render_latex_native(gt_latex, save_dir=save_dir, display_mode=display_mode)
        
        if img_pred is None:
            logger.error(f"Failed to render predicted: {pred_latex}")
        if img_gt is None:
            logger.error(f"Failed to render ground truth: {gt_latex}")
        
        if img_pred is None or img_gt is None:
            return {
                "mse": float('inf'),
                "ssim": 0.0,
                "normalized_distance": 1.0,
                "render_success": False
            }
        
        metrics = compute_image_similarity(img_pred, img_gt)
        metrics["render_success"] = True
        return metrics
        
    except Exception as e:
        logger.error(f"Error in latex_image_loss: {e}")
        return {
            "mse": float('inf'),
            "ssim": 0.0,
            "normalized_distance": 1.0,
            "render_success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug native LaTeX rendering.")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save rendered images")
    parser.add_argument("--display-mode", type=str, choices=["inline", "display"], default="inline",
                        help="Rendering mode: inline uses $...$, display uses \\[...\\]")
    args = parser.parse_args()
    
    examples = [
        ("\\frac{1}{2}", "\\dfrac{1}{2}"),
        ("x^2 + 2x + 1", "(x+1)^2"),
        ("\\int_{0}^{\\infty} e^{-x^2} dx", "\\int_0^\\infty e^{-x^2} dx"),
        ("\\sum_{i=1}^{n} i", "\\sum_{i=1}^n i"),
        ("114514", "191919810"),
        ("[123]", "\\left[123\\right]"),
        ("x^{y^z}", "{x^y}^z"),
        ("\\int_{0}^{\\infty} \\, e^{-x^2} dx", "\\int_0^\\infty e^{-x^2} dx"),
        ("\\begin{pmatrix}1 & 2 \\\\ 3 & 4\\end{pmatrix}", "\\begin{bmatrix}1 & 2 \\\\ 3 & 4\\end{bmatrix}"),
    ]
    
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
        
        result_native = latex_image_loss(ex1, ex2, save_dir=args.save_dir, display_mode=args.display_mode)
        
        mse_val = result_native.get('mse')
        if mse_val == float('inf'):
            mse_str = "inf"
        elif isinstance(mse_val, (int, float)):
            mse_str = f"{mse_val:.4f}"
        else:
            mse_str = str(mse_val)
            
        ssim_val = result_native.get('ssim', 0)
        ssim_str = f"{ssim_val:.4f}" if isinstance(ssim_val, (int, float)) else str(ssim_val)
        
        print(f"Native LaTeX renderer [{args.display_mode} mode]: MSE = {mse_str}, SSIM = {ssim_str}")
        print("---")