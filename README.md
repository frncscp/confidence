# confidence

### For Positive Classes:
$C_{k}^{+} = \frac{(S * {S}')+(P * {P}')}{{S}'+{P'}}$

### For Negative Classes:
$C_{k}^{-} = \frac{(S * {S}')+((1-P) * {P}')}{{S}'+{P}'}$

with a range of **[0, 1)**

## **Where:**

**S** represents normalized score (+1 for each image correctly classified, -1 if viceversa) from 0 to 1.

**P** represents average prediction.

**S´** and **P´** are each variable weights.

**k** is the classification threshold used to get the score.

## **How it works:**

Each score and average prediction is extracted from a folder with all positive or all negative class images. 

There's a Tensorflow implementation in this repo that iterates through all models and folders (divided in positive and negative classes), saves the results on a dictionary, and shows them on a graph

The general idea is that not only what percentage of the data was correctly classified matters, but also *how confident the model was to give the right answer*. Two different models that give apparently similar results (specially when dealing with a small dataset) could have entirely different probability distributions, and one of them could have a median closer to the classification threshold than the other (which makes the latter preferable and gives and idea of how it could behave with data not seen during training).
