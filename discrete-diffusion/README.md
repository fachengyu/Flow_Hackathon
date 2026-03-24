# Discrete Diffusion Workshop

Walkthroughs of two approaches to generative modeling over discrete sequences.

## MDLM — Masked Diffusion Language Models

A masked diffusion model that generates fixed-length sequences by iteratively unmasking tokens.

- `mdlm_walkthrough_exercise.ipynb` — notebook with exercises to fill in
- `mdlm_walkthrough_soln.ipynb` — completed solution notebook
- `mdlm_data.py` — data utilities: text8 download, tokenization, and evaluation metrics
- `mdlm_model.py` — DiT-style transformer backbone

## EditFlow — Flow Matching with Edit Operations

An extension that generates variable-length sequences via insertion operations, modeled as a continuous-time Markov chain (CTMC). Addresses limitations of fixed-length masked diffusion.

- `editflow_walkthrough.ipynb` — self-contained walkthrough notebook
- `ef_data.py` — data utilities for EditFlow (text8 with BOS/EOS tokens)

## Instructions

Work through each exercise notebook and refer to the solution notebook as needed.
