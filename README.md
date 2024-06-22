# The joint of Contrastive Preference Optimization (CPO) & Simple Preference Optimization (SimPO)

This repository contains the code and released models for [CPO](https://arxiv.org/pdf/2401.08417) and [SimPO](https://arxiv.org/abs/2405.14734). The code is based on [SimPO github](https://github.com/princeton-nlp/SimPO). We focus on highlighting reference-free preference learning and demonstrating the effectiveness of SimPO. 

Additionally, we integrate length normalization and target reward margin into CPO, showing promising results and the poential benefits to combine them together. 

## Introduction
CPO and SimPO share similar objectives but have different goals. CPO adds a BC-regularizer to prevent the model from deviating too much from the preferred data distribution.

$L_{CPO}(\pi_\theta;U) = E_{(x,y_w,y_l) \sim \mathcal{D}} \Big[ \log \sigma \Big( \beta \log \pi_{\theta}(y_w | x)  - \beta \log \pi_{\theta}(y_l | x) \Big) - \log \pi_\theta(y_w| x)\Big]$

SimPO incorporates length normalization and target reward margin to improve model performance and prevent the generation of long but low-quality sequences:

$L_{SimPO}(\pi_\theta;U) = E_{(x,y_w,y_l) \sim \mathcal{D}} \Big[ \log \sigma \Big( \frac{\beta}{|y_w|} \log \pi_{\theta}(y_w | x)  - \frac{\beta}{|y_l|} \log \pi_{\theta}(y_l | x)  - \gamma  \Big) \Big]$

These two objectives can be jointly used, which we call CPO-SimPO:

$L_{CPO-SimPO}(\pi_\theta;U) = E_{(x,y_w,y_l) \sim \mathcal{D}} \Big[ \log \sigma \Big( \frac{\beta}{|y_w|} \log \pi_{\theta}(y_w | x)  - \frac{\beta}{|y_l|} \log \pi_{\theta}(y_l | x)  - \gamma  \Big)- \alpha \log \pi_\theta(y_w| x)\Big]$

## Released Models
Below is the list of models that we evaluated .

| models                       |                                                                                                           | AE2 LC | AE2 WR |
|------------------------------|-----------------------------------------------------------------------------------------------------------|:------:|:------:|
| Llama3 Instruct 8B SimPO (reported)     | [haoranxu/Llama-3-Instruct-8B-SimPO](https://huggingface.co/haoranxu/Llama-3-Instruct-8B-SimPO) |  44.7  |  40.5  |
| Llama3 Instruct 8B CPO       | [haoranxu/Llama-3-Instruct-8B-CPO](https://huggingface.co/haoranxu/Llama-3-Instruct-8B-CPO) |  36.07  |  40.06  |
| Llama3 Instruct 8B CPO-SimPO | [haoranxu/Llama-3-Instruct-8B-CPO-SimPO](https://huggingface.co/haoranxu/Llama-3-Instruct-8B-CPO-SimPO) |  46.94  |  44.72  |

## Training Scripts
* Llama3-Instruct with CPO + SimPO:
```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_cpo.py training_configs/llama-3-8b-instruct-cpo-simpo.yaml
```
## Others
For environment settings and evaluation steps, please refer to the original [SimPO github](https://github.com/princeton-nlp/SimPO).

## Citation
```bibtex
@inproceedings{
xu2024contrastive,
title={Contrastive Preference Optimization: Pushing the Boundaries of {LLM} Performance in Machine Translation},
author={Haoran Xu and Amr Sharaf and Yunmo Chen and Weiting Tan and Lingfeng Shen and Benjamin Van Durme and Kenton Murray and Young Jin Kim},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=51iwkioZpn}
}
```
```bibtex
@article{meng2024simpo,
  title={{SimPO}: Simple Preference Optimization with a Reference-Free Reward},
  author={Meng, Yu and Xia, Mengzhou and Chen, Danqi},
  year={2024}
}
```
