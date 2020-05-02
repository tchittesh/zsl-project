## Introduction

This repository contains implementations of 5 classical zero-shot algorithms in the usual as well as the Generalized zero-shot settings using the 
`Proposed Split` and evaluation protocols outlined in 
[Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly](https://arxiv.org/abs/1707.00600) (**ZSLGBU**) by Yongqin Xian, Christoph H. Lampert, Bernt Schiele, Zeynep Akata (TPAMI 2018).

To the best of my knowledge, this is the **first public implementation** of SAE, ALE, SJE and DeViSE under the ZSLGBU protocol. An existing implementation of `ESZSL` can be found [here](https://github.com/sbharadwajj/embarrassingly-simple-zero-shot-learning). To this, I have also added the Generalized ZSL functionality.

## Reference Papers

The original papers corresponding to the 5 algorithms are:

[1] SAE - [Semantic Autoencoder for Zero-Shot Learning](https://arxiv.org/abs/1704.08345).
Elyor Kodirov, Tao Xiang, Shaogang Gong.
CVPR, 2017.

[2] ALE - [Label-Embedding for Image Classification](https://arxiv.org/abs/1503.08677).
Zeynep Akata, Florent Perronnin, Zaid Harchaoui, Cordelia Schmid.
TPAMI, 2016.

[3] SJE - [Evaluation of Output Embeddings for Fine-Grained Image Classification](https://arxiv.org/abs/1409.8403).
Zeynep Akata, Scott Reed, Daniel Walter, Honglak Lee, Bernt Schiele.
CVPR, 2015.

[4] ESZSL - [An embarrassingly simple approach to zero-shot learning](http://proceedings.mlr.press/v37/romera-paredes15.pdf).
Bernardino Romera-Paredes, Philip H. S. Torr.
ICML, 2015.

[5] DeViSE - [DeViSE: A Deep Visual-Semantic Embedding Model](http://papers.nips.cc/paper/5204-devise-a-deep-visual-semantic-embedding-model.pdf).
Andrea Frome*, Greg S. Corrado*, Jonathon Shlens*, Samy Bengio, Jeffrey Dean, Marc’Aurelio Ranzato, Tomas Mikolov.
NIPS, 2013. 

## Setup

```
git clone https://github.com/mvp18/Popular-ZSL-Algorithms.git
cd Popular-ZSL-Algorithms
bash setup.sh
```

This downloads data (splits, Res101 features and class embeddings) corresponding to the `Proposed Split` for AWA1, AWA2, CUB, SUN and aPY. To know more about the individual files, refer to the `README.txt` file available inside `xlsa17` folder.