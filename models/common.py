# -*- coding: utf-8 -*-
"""
models/common.py
This script sets up a series of data extraction models using the dspy library for pathology reports. Common model includes basic dspy functionality, cancer examination, and json handling. It includes model loading, signature definitions for various cancer types, and functions to convert model predictions into structured JSON formats.

author: Hong-Kai (Walther) Chen, Po-Yen Tzeng and Kai-Po Chang @ Med NLP Lab, China Medical University
date: 2025-10-13
"""
__version__ = "1.0.0"
__date__ = "2025-10-13"
__author__ = ["Hong-Kai (Walther) Chen", "Po-Yen Tzeng", "Kai-Po Chang"]
__copyright__ = "Copyright 2025, Med NLP Lab, China Medical University"
__license__ = "MIT"

import dspy
from typing import Literal

model_list = {"gemma4b": "ollama_chat/gemma3:4b",
          "gemma1b": "ollama_chat/gemma3:1b",
          "med8b": "ollama_chat/thewindmom/llama3-med42-8b",
          "gemma12b": "ollama_chat/gemma3:12b", 
          "gemma27b": "ollama_chat/gemma3:27b", 
          "med70b": "ollama_chat/thewindmom/llama3-med42-70b",
          "gpt": "ollama_chat/gpt-oss:20b",
          "phi4": "ollama_chat/phi4",
          "qwen30b": "ollama_chat/qwen3:30b"
}

localaddr = "http://localhost:11434"

def load_model(model_name: str):
    if model_name not in model_list:
        raise ValueError(f"Model {model_name} not found. Available models: {list(model_list.keys())}")

    model = model_list[model_name]
    lm = dspy.LM(
        model=model,
        api_base=localaddr,
        api_key="",
        model_type="chat",
        top_p=0.7,
        max_tokens=16384,
        num_ctx=16384,
        temperature=0.7,
        seed = 10
    )
    print(f"Loaded model: {model_name}")
    return lm

# 2 . define classes and set up Signatures

def autoconf_dspy (model_name: str):
    lm = load_model(model_name)
    dspy.configure(lm=lm)

class is_cancer(dspy.Signature):
    """You are a cancer registrar, you need to identify whether or not this report belongs to PRIMARY cancer excision eligible for cancer registry, and if so, which organ the cancer belongs to. If no viable tumor is present after excision, you should not register this case. If only carcinoma in situ or high-grade dysplasia, you should not register this case."""
    
    report: list = dspy.InputField(desc = 'this is a pathologic report, separated into paragraphs. you should determine whether or not this report belongs to cancer excision eligible for cancer registry')

    cancer_excision_report: bool = dspy.OutputField(desc= 'identify whether or not this report belongs to PRIMARY cancer excision eligible for registry for cancer excision. If no viable tumor is present after excision, you should not register this case. If only carcinoma in situ or high-grade dysplasia, you should not register this case.')#a point
    #exp:
    cancer_category: Literal['stomach','colorectal','breast','esophagus', 'lung', 'prostate', "thyroid", "pancreas", "cervix", "liver", "others"]|None = dspy.OutputField(desc = 'identify which organ the primary cancer arises from. Currently only ten are implemented, if it IS a cancer excision report, but primary site not included in these standard organs, should be classified as others.')
    cancer_category_others_description: str|None = dspy.OutputField(desc = 'if is cancer_excision report AND cancer_category is others, please specify the organ here. if not, return null.')

class ReportJsonize(dspy.Signature):
    """You are cancer registrar, and you are assigned a task to manually convert the raw pathology report into a roughly structured json format. Keep original wording as much as possible. Try to follow the order of cancer checklists."""
    report: list = dspy.InputField(desc = 'this is a raw pathological report, separated into paragraphs. You need to convert it into a roughly structured json format, keeping original wording as much as possible.')
    cancer_category: Literal['stomach','colorectal','breast','esophagus', 'lung', 'prostate', "pancreas", "thyroid", "cervix", "liver"]|None = dspy.InputField(desc = 'which part the cancer belongs to. You need to convert it into a roughly structured json format, keeping original wording as much as possible.')
    output: dict = dspy.OutputField(desc = 'You are cancer registrar, and you are assigned a task to manually convert the raw pathology report into a roughly structured json format. Keep original wording as much as possible. Try to follow the order of cancer checklists.')
