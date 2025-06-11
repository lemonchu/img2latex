import os
import json
import random
from tqdm import tqdm
from jinja2 import Template
from datasets import load_dataset
from transformers import AutoProcessor
# 新增依赖：LaTeX 渲染及相似度计算
from latex_image_loss import latex_image_loss  # 假设已将该模块放置在 PYTHONPATH 中

instruction = 'Convert the equation images to LaTeX equations.'

def convert_to_conversation(sample):
    """Convert a single dataset sample to a conversation (chat) format."""
    prompt=instruction
    image=sample['image']
    text=sample['text']
    return {'prompt':prompt, 'image':image,'answer':text}


def _check_is_passed(problem):
    response = problem['response']
    if problem.get('finish_reason') != 'stop':
        return 0
    if '\\boxed{' not in response or '}' not in response:
        return 0
    pred = response.rsplit('\\boxed{', 1)[-1].rsplit('}', 1)[0]
    gt = problem.get('answer', '').strip()
    metrics = latex_image_loss(pred, gt, render_method=os.getenv('LATEX_RENDER', 'matplotlib'))
    
    return 1-metrics["ssim"]


def _postprocess_problem(problem):
    problem['reward'] = int(_check_is_passed(problem))
    return problem


def _postprocess(problems):
    results = []
    for problem in tqdm(problems, desc="Postprocessing"):
        results.append(_postprocess_problem(problem))
    return results


class TrainDataset:
    def __init__(self, data_path: str, samples_per_iteration: int, rollouts_per_sample: int):
        self.samples_per_iteration = samples_per_iteration
        self.rollouts_per_sample = rollouts_per_sample
        self.dataset = load_dataset('unsloth/LaTeX_OCR', split='train[:3000]')
        self.data=[convert_to_conversation(s) for s in tqdm(self.dataset, desc='Converting Train')]

    def preprocess(self):
        sampled = random.sample(self.data, self.samples_per_iteration)
        problems = []
        for item in sampled:
            prompt = item['prompt']
            problem={
                'prompt': prompt,
                'image': item['image'],
                'answer': item['answer'],
                'sampling_params': {
                    'temperature': 1.0,
                    'stop': ['</answer>'],
                }
            }
            problems.extend([problem]*self.rollouts_per_sample)
           
        return problems

    def postprocess(self, problems):
        return _postprocess(problems)


class EvalDataset:
    def __init__(self, data_path: str, samples_per_iteration: int, rollouts_per_sample: int):
        self.samples_per_iteration = samples_per_iteration
        self.rollouts_per_sample = rollouts_per_sample
        self.dataset = load_dataset('unsloth/LaTeX_OCR', split='test[:500]')
        self.data=[convert_to_conversation(s) for s in tqdm(self.dataset, desc='Converting Eval')]

    def preprocess(self):
        sampled = random.sample(self.data, self.samples_per_iteration)
        problems = []
        for item in sampled:
            prompt = item['prompt']
            problem={
                'prompt': prompt,
                'image': item['image'],
                'answer': item['answer'],
                'sampling_params': {
                    'temperature': 1.0,
                    'stop': ['</answer>'],
                }
            }
            problems.extend([problem]*self.rollouts_per_sample)
           
        return problems

    def postprocess(self, problems):
        return _postprocess(problems)
