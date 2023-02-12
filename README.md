# efficiency
**Is described as:**

-For Positive Classes:
$E = \frac{(S * {S}')+(P * {P}')}{{S}'+{P}'}$

-For Negative Classes:
$E = \frac{(S * {S}')+(P * {P}')}{{S}'+{P}'}$

**Where:**

S represents normalized score (+1 for each image correctly classified, -1 if viceversa) from 0 to 1

P represents average prediction

S´ and P´ are each variable biases

**How it works:**

Each score and average prediction is extracted from a folder with all positive or all negative class images. 

There's an implementation in this repo (which is a bit slow) that iterates through all models and folders (divided in positive and negative classes), saves the results on a dictionary, and shows them on a graph
