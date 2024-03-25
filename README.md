# FrameChain

LangChain for large vision models (LVMs)
FrameChain is a library for composing chains of transformations on images. It is inspired by the utility of large vision models (LVMs) -- models that perform learned transformations on images -- and is designed to be used in conjunction with them.

## Coming from LangChain

If you've already used LangChain, FrameChain is very similar. The main difference is that FrameChain is designed to work with images and LVMs, while LangChain is designed to work with text and LLMs.

# TODO: make this a mermain compare and contrast diagram
# TODO: add more points to this diagram
- LLMs receive a sequence of tokens
- LVMs receive a sequence of images
- LangChain uses string templates to structure text prompts
- FrameChain uses spatial templates to structure image prompts
- LLMs are semantic kernels
- LVMs are visual kernels

# Getting Started

`pip install framechain`

# Usage

show some examples

# Ack

- [Sequential Modeling Enables Scalable Learning for Large Vision Models](https://yutongbai.com/lvm.html)
    @misc{bai2023sequential,
      title={Sequential Modeling Enables Scalable Learning for Large Vision Models}, 
      author={Yutong Bai and Xinyang Geng and Karttikeya Mangalam and Amir Bar and Alan Yuille and Trevor Darrell and Jitendra Malik and Alexei A Efros},
      year={2023},
      eprint={2312.00785},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }

- [Data-efficient Large Vision Models through Sequential Autoregression](https://arxiv.org/pdf/2402.04841.pdf) 
    @misc{guo2024dataefficient,
      title={Data-efficient Large Vision Models through Sequential Autoregression}, 
      author={Jianyuan Guo and Zhiwei Hao and Chengcheng Wang and Yehui Tang and Han Wu and Han Hu and Kai Han and Chang Xu},
      year={2024},
      eprint={2402.04841},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }
