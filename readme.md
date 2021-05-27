# orf: ordered random forests

Welcome to the repository of the Python implementation for the Ordered Forest estimator
([Lechner & Okasa, 2019](https://arxiv.org/abs/1907.02436)) for machine learning estimation
of the ordered choice models. The current Python implementation is focused on the prediction
exercise and does not yet provide the procedures for statistical inference. For the full
functionality please refer to the `R` package `orf` available on [CRAN](https://CRAN.R-project.org/package=orf)
repository.

## Introduction

This repository provides the Python implementation of the Ordered Forest estimator
as developed in [Lechner and Okasa (2019)](https://arxiv.org/abs/1907.02436).
The Ordered Forest flexibly estimates the conditional probabilities of models with
ordered categorical outcomes (so-called ordered choice models). Additionally to
common machine learning algorithms the Ordered Forest provides functions for estimating
marginal effects and thus provides similar output as in standard econometric models
for ordered choice. The core Ordered Forest algorithm relies on the random forest
implementation from the `scikit-learn` module ([Pedregosa et al., 2011](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)).
The here provided functions for estimating the Ordered Forest also use the `scikit-learn`
typical command syntax and should thus be easy to use.

## Installation

The implementation of the Ordered Forest relies on Python 3 and requires `scikit-learn`
as well as `numpy` and `pandas`. The required modules can be installed by navigating to the root
of this project and executing the following command: `pip install -r dependencies.txt`.

## Examples

The example below demonstrates the basic functionality of the Ordered Forest.

```python
# load the Ordered Forest
from orf.orf import OrderedForest

# import additonal modules
import pandas as pd

# read in example data from the orf package in R
odata = pd.read_csv('orf/odata.csv')

# define outcome and features
outcome = odata['Y']
features = odata.drop('Y', axis=1)

# Ordered Forest estimation

# initiate the class with tuning parameters
oforest = OrderedForest(n_estimators=1000, min_samples_leaf=5, max_features=0.3)
# fit the model
oforest.fit(X=features, y=outcome)
# predict ordered probabilities
oforest.predict(X=features)
# evaluate the prediction performance
oforest.performance()
# evaluate marginal effects
oforest.margin(X=features)
```

The complete example code as well as the example data are available in the `orf` folder.

## References

- Lechner, M., & Okasa, G. (2019). Random Forest Estimation of the Ordered Choice Model. arXiv preprint arXiv:1907.02436. <https://arxiv.org/abs/1907.02436>
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, pp. 2825-2830. <https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html>

Special thanks goes to [JLDC](https://github.com/JLDC).
