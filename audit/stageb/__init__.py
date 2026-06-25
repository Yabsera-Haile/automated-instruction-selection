"""Stage B: does the Stage-A selection distortion cause downstream harm?

Fine-tune a target model (Qwen2.5-7B base) on a decisive subset of the Stage-A
selections, evaluate per-language and per-skill, and test whether practitioner-default
selectors (perplexity-low, quality) yield competitive average accuracy but degraded
low-resource-language / worst-skill accuracy vs random and full-data.

Execution split: [LOCAL] steps (subset materialization, eval-language list, analysis)
run on the 4GB dev box; [SERVER] steps (7B LoRA training + evaluation) run on the GPU
server and are confirmed only from pasted output.
"""
