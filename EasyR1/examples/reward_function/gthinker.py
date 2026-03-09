import argparse
import re
import os
from mathruler.grader import grade_answer
# from math_verify import verify, parse
from typing import Optional, Any
# from math_verify.errors import TimeoutException
# from math_verify.metric import math_metric
# from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig,StringExtractionConfig
import ast
import json
from tqdm import tqdm
from sympy import simplify, Tuple
from sympy.parsing.latex import parse_latex
import traceback
from collections import Counter
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

REWARD_NAME = "gthinker"
REWARD_TYPE = "sequential_diff" #ZYF Modify: support different reward type

def preprocess_tuple(expr):
    # 删除 \left 和 \right
    expr = expr.replace(r'\left', '').replace(r'\right', '')
    # 查找形式如 (a, b) 并转为 Tuple(a, b)
    expr = re.sub(r'\(\s*([^\(\),]+?)\s*,\s*([^\(\),]+?)\s*\)', r'Tuple(\1, \2)', expr)
    return expr


def safe_parse(expr):
    try:
        return simplify(parse_latex(expr, strict=False))
    except Exception as e:
        print(f"无法解析: {expr} -> {e}")
        return None


def format_reward(predict_str: str, is_pra = True) -> float:
    if is_pra:
        pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    else:
        pattern = re.compile(r"<answer>.*?</answer>\s*<think>.*?</think>", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0

def math_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    try:
        matches = re.findall(r'<answer>(.*?)</answer>', predict_str, re.DOTALL)
        given_answer = matches[-1].strip()
    except Exception as e:
        return 0.0
    
    try:
        ground_truth_ = ground_truth.strip('$')
        given_answer_ = given_answer.strip('$')
        return 1.0 if grade_answer(given_answer_, ground_truth_) else 0.0
    except:
        return 0.0
    

def content_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    """
    For short ones
    """
    try:
        ground_truth = ground_truth.strip()
        content_match = re.search(r"<answer>(.*?)</answer>", predict_str)
        given_answer = content_match.group(1) if content_match else predict_str
        if "<answer>" in ground_truth:
            ground_truth = re.search(r"<answer>(.*?)</answer>", ground_truth).group(1).strip()
        if given_answer == ground_truth:
            return 1.0
    except Exception:
        pass
    return 0.0


def edit_distance_reward(predicted_str, ground_truth_str, threshold=0.6, reward_type='similarity'):
    """For long ones"""
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # 插入、删除、替换的代价
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

    def normalized_edit_distance(s1, s2):
        edit_dist = levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        
        if max_len == 0:
            return 0.0
        
        return edit_dist / max_len

    def edit_distance_similarity(s1, s2):
        return 1 - normalized_edit_distance(s1, s2)

    if reward_type == 'similarity':
        similarity = edit_distance_similarity(predicted_str, ground_truth_str)
        if similarity > threshold:
            return similarity
        else:
            return 0.0
    elif reward_type == 'distance':
        norm_distance = normalized_edit_distance(predicted_str, ground_truth_str)
        if norm_distance < threshold:  
            return 1 - norm_distance  
        else:
            return 0.0
    else:
        raise ValueError("reward_type must be 'similarity' or 'distance'")


def gthinker_sys_compute_score(predict_str: str, ground_truth: str, domain: str, qtype: str, equ_tag: bool=False, format_weight: float = 0.8, is_pra = True) -> float:

    if domain in {'Math'}:
        acc_score = math_accuracy_reward(predict_str, ground_truth)
    else:
        pattern = r'^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$'
        if re.fullmatch(pattern, ground_truth.strip()) is not None:  # 精确数字匹配
            acc_score = content_accuracy_reward(predict_str, ground_truth)
        else:
            if len(ground_truth.split(' ')) > 5:
                acc_score = edit_distance_reward(predict_str, ground_truth)
            else:
                acc_score = content_accuracy_reward(predict_str, ground_truth)


    format_score = format_reward(predict_str, is_pra)
    #score =  0.5 * acc_score + 0.5 * format_score
    try:
        score =  (1-format_weight) * acc_score + format_weight * format_score
    except:
        raise AssertionError(f"Params Error: Domain: {domain}, qtype: {qtype}, equ_tag: {equ_tag}")
    return score, acc_score, format_score

def compute_score(reward_input: dict[str: Any], format_weight: float = 0.5) -> dict[str, float]:
    predict_str = reward_input["response"]
    ground_truth = reward_input["ground_truth"]
    domain = reward_input.get("domains", "Others")
    qtype = reward_input["type"]
    if qtype == "geoqa+":
        qtype = "FIB"
        domain = "Math"
    else:
        qtype = "MCQ"
    equ_tag = reward_input.get("equ_tag", False)
    score, acc_score, format_score = gthinker_sys_compute_score(predict_str, ground_truth, domain, qtype, equ_tag, format_weight)
    return {"overall": score, "format": format_score, "accuracy": acc_score}
