# COMS4995 Geometric Data Analysis Final Project

The goal of this project is to establish a better framework for analyzing neural time-series data. Standard neuroscience analysis uses UMAP, t-SNE, PCA, or other unsatisfying projections with an axiomatic assumption that the underlying activity space is 2 or 3 dimensions (either the task is very simple so there are only 2 or 3 relevant features for neurons to encode, or the authors would like a nice pretty figure, or both).  We aim to root this assumption more rigorously by demonstrating that the relevant topological features of neural activity remain dominant in unprojected data.

We begin with a small improvement on standard simplicial complex construction. Given an $\epsilon$ feature scale, we construct a standard simplicial complex. Then, we add edges from datapoints $x_{t}^i$ to $x_{t+1}^i$, where $i$ denotes the $i$-th trial such that $x_{t}^i$ represents the $t$-th datapoint in trial $i$. Given $n$ neurons, we can say that $x_{t}^i \in \R^n$. There are 2 methods of comparison here. We will call our method TrajectoryMap. We thus create a control group for analysis (i.e. standard PCA, IsoMap, and naive simplicial complex) and an experimental group for analysis (i.e. TrajectoryMap)

## Evaluation Methods
1) Betti number calculation: Friedman 1995 shows that Betti numbers are computable algorithmically given a simplicial complex and that the Betti number corresponding to the highest dimension of boundary map corresponds to the homology group of the complex. We aim to compute the Betti numbers on our dataset and see if they hold rigorously across trials.
2) Spectral clustering: we aim to use standard spectral clustering methods like IsoMap on a training set of data and see how well the control analysis does on predicting datapoints compared to TrajectoryMap.
3) Epsilon tightening: we hypothesize that TrajectoryMap allows us to achieve the same level of prediction accuracy with a lower $\epsilon$ treshold, in some sense allowing us to eliminate noise with much less loss of information.

## Key Assumptions
1) The geometric structure of neural trajectories in high dimensions gives us useful information about the structure of the behavior studied
2) Homology remains invariant under homotopy for a given experiment regardless of the sample studied

Note that we do not need to consider smoothness of underlying activity space as we are studying the simplicial complexes! We also hope to justify the second assumption by showing that activity space is indeed well-defined by its homology. 

## Datasets
1) Eye saccade data: [NEED TO UPDATE ON SPECIFICS]
2) Wheel rotation data: [NEED TO UPDATE ON SPECIFICS]

We hypothesize that TrajectoryMap is more effective on tasks that require very time-dependent tasks, like motor behavior!

Thus, there are a multitude of comparatives we need to look at.
## Experimental Procedure
1) Select one of four models: PCA, IsoMap, naive simplicial complex, TrajectoryMap complex
2) Select the appropriate evaluation metric
    a) For Betti number calculation, we are only interested in comparing the naive simplicial complex with the TrajectoryMap complex, and no training/test split is needed. 
    b) For spectral clustering, we are interested in all four methods and need to evaluate on a training/test split.
    c) For epsilon tightening, we are interested in comparing IsoMap vs. naive simplicial complex vs. TrajectoryMap complex. Here as well we need a training/test split.
3) Select the dataset we evaluate on!
