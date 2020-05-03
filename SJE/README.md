## Running the code

#### Train

```
python sje/sje_gzsl.py -data AWA2/AWA1/CUB/SUN/APY -e [EPOCHS] -es [EARLY STOP] -norm [NORMALIZATION TYPE] -lr [LEARNING RATE] -mr [SVM LOSS MARGIN]
```
For testing, set learning rate (lr), margin (mr), and normalization type (norm) to best combination from the tables below.

## Results

| Dataset | ZSLGBU Results| Repository Results | Hyperparams from Val     |
|---------|:-------------:|:------------------:|:------------------------:|
| aPY     |     32.9      |      32.86         |lr=0.01, mr=1.5, norm=None|
| SUN     |     53.7      |      53.47         |lr=1.0, mr=2.0, norm=std  |      |

#### Generalized ZSL

To be updated soon...

### References

[1] [Original C++ Code by Authors](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/evaluation-of-output-embeddings-for-fine-grained-image-classification/)