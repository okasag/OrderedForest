"""
orf: Ordered Random Forests.

Python implementation of the Ordered Forest as in Lechner & Okasa (2019).

Showcase application of the Ordered Forest estimator.

"""

# import modules
import pandas as pd

# load the ordered forest
from orf import OrderedForest

# read in example data from the orf package in R
odata = pd.read_csv('odata.csv')

# define outcome and features
outcome = odata['Y']
features = odata.drop('Y', axis=1)

# Ordered Forest estimation

# initiate the class with tuning parameters
oforest = OrderedForest(n_estimators=500, min_samples_leaf=5, max_features=0.3)
# fit the model
oforest.fit(X=features, y=outcome)
# predict ordered probabilities
oforest.predict(X=features)
# predict ordered classes
oforest.predict(X=features, prob=False)
# evaluate the prediction performance
oforest.performance()
# evaluate marginal effects
oforest.margin(X=features)

# end fo the example
