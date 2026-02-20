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
    @inproceedings{yuviler2026expairtllm,
      title     = {ExPairT-LLM: Exact Learning for LLM Code Selection by Pairwise Queries},
      author    = {Yuviler, Tom and Drachsler-Cohen, Dana},
      booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
      year      = {2026},
      note      = {AAAI 2026},
      url       = {https://arxiv.org/abs/2511.10855}
    }
