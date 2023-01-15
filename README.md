# Low Rank Adapters (LoRA)

Rudimentary codebase for adding Low Rank Adapters to transformer models, based on the paper, [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).

We inject trainable rank decomposition matrices into layers of the transformer. If a model is to large to train this is a well-performing alternative.

### BibTeX
```
@article{DBLP:journals/corr/abs-2106-09685,
  author    = {Edward J. Hu and
               Yelong Shen and
               Phillip Wallis and
               Zeyuan Allen{-}Zhu and
               Yuanzhi Li and
               Shean Wang and
               Weizhu Chen},
  title     = {LoRA: Low-Rank Adaptation of Large Language Models},
  journal   = {CoRR},
  volume    = {abs/2106.09685},
  year      = {2021},
  url       = {https://arxiv.org/abs/2106.09685},
  eprinttype = {arXiv},
  eprint    = {2106.09685},
  timestamp = {Tue, 29 Jun 2021 16:55:04 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2106-09685.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```"# low-rank-adapters" 
