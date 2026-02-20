# ExPairT-LLM

Official code for **ExPairT-LLM: Exact Learning for LLM Code Selection by Pairwise Queries** (**AAAI 2026**).  
Paper (arXiv): https://arxiv.org/abs/2511.10855

## Overview
ExPairT-LLM selects a program from a set of LLM-generated candidate solutions using **pairwise membership** and **pairwise equivalence** queries, refining candidates with generated tests.

## Setup
    git clone https://github.com/TomYuviler/ExPairT-LLM.git
    cd ExPairT-LLM
    pip install -r requirements.txt

Set API keys via environment variables (recommended), e.g.:
    export OPENAI_API_KEY="..."

## Run
    python ExPairT.py --dataset_name mbpp_sanitized --num_candidate 8 \
      --llm_for_generation o1-mini --llm_for_inputs o1-mini --llm_for_oracle o1-mini

## Citation
    @misc{yuviler2025expairtllm,
      title        = {ExPairT-LLM: Exact Learning for LLM Code Selection by Pairwise Queries},
      author       = {Tom Yuviler and Dana Drachsler-Cohen},
      year         = {2025},
      eprint       = {2511.10855},
      archivePrefix= {arXiv},
      primaryClass = {cs.LG},
      doi          = {10.48550/arXiv.2511.10855}
    }
