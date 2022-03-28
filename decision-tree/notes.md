## Lesson Outline
* Recommendation applications
* Entropy
* Information gain
* Hyperparameters
* Decision trees in sklearn


### Entropy
Entropy is inversly proportinal to knowledge. The less knowledge one has, the more entropy. Entropy is a measure to decide which set consists of most similar data points.

When the output is of two types containing m in class1 and n no of datapoints in class2

Entropy = - [(m/m+n)*log<sub>2</sub>(m/m+n) + (n/m+n)*log<sub>2</sub>(n/m+n)]

#### Multi-class Entropy

Entropy = - [p<sub>1</sub>log(p<sub>1</sub>) + p<sub>2</sub>log(p<sub>2</sub>) + p<sub>3</sub>log(p<sub>3</sub>) + .... + p<sub>n</sub>log(p<sub>n</sub>)]

#### Quiz
1. What is the entropy for a bucket with a ratio of four red balls to ten blue balls? Input your answer to at least three decimal places.
Entropy = -[4/14*log<sub>2</sub>(4/14) + 10/14*log<sub>2</sub>(10/14)]

2. If we have a bucket with eight red balls, three blue balls, and two yellow balls, what is the entropy of the set of balls?

Entropy = -[8/13*log<sub>2</sub>(8/13) + 3/13*log<sub>2</sub>(3/13) + 2/13*log<sub>2</sub>(2/13)]

### Information Gain

Information Gain = Entropy(parent set) - avg(entropy of children set)


### Relation
Ptobability -> Entropy -> Knowledge Gain 


### Hyperparameters
In order to create decision trees that will generalize to new problems well, we can tune a number of different aspects about the trees. We call the different aspects of a decision tree "hyperparameters"

1. Maximum Depth - The largest possible length between the root to a leaf. A tree of maximum length
2. Minimum number of samples to split - A node must have at least ___ # in order to be large enough to split.
3. Minimum number of samples per leaf - At least ___ # samples we allow on each leaf

**Important points**
1. Large depth very often causes overfitting, since a tree that is too deep, can memorize the data. Small depth can result in a very simple model, which may cause underfitting. 
2. Small minimum samples per split may result in a complicated, highly branched tree, which can mean the model has memorized the data, or in other words, overfit. Large minimum samples may result in the tree not having enough flexibility to get built, and may result in underfitting.