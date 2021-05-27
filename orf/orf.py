"""
orf: Ordered Random Forests.

Python implementation of the Ordered Forest as in Lechner & Okasa (2019).

Definitions of class and functions.

"""

# import modules
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# define OrderedForest class
class OrderedForest:
    """
    Ordered Random Forests class labeled 'OrderedForest'.

    includes methods to fit the model, predict and estimate marginal effects.

    Parameters
    ----------
    n_estimators : TYPE: integer
        DESCRIPTION: Number of trees in the forest. The default is 1000.
    min_samples_leaf : TYPE: integer
        DESCRIPTION: Minimum leaf size in the forest. The default is 5.
    max_features : TYPE: float
        DESCRIPTION: Share of random covariates (0,1). The default is 0.3.

    Returns
    -------
    None. Initializes parameters for Ordered Forest.
    """

    # define init function
    def __init__(self, n_estimators=1000,
                 min_samples_leaf=5,
                 max_features=0.3):

        # check and define the input parameters
        # check the number of trees in the forest
        if isinstance(n_estimators, int):
            # check if its at least 1
            if n_estimators >= 1:
                # assign the input value
                self.n_estimators = n_estimators
            else:
                # raise value error
                raise ValueError("n_estimators must be at least 1"
                                 ", got %s" % n_estimators)
        else:
            # raise value error
            raise ValueError("n_estimators must be an integer"
                             ", got %s" % n_estimators)

        # check if minimum leaf size is integer
        if isinstance(min_samples_leaf, int):
            # check if its at least 1
            if min_samples_leaf >= 1:
                # assign the input value
                self.min_samples_leaf = min_samples_leaf
            else:
                # raise value error
                raise ValueError("min_samples_leaf must be at least 1"
                                 ", got %s" % min_samples_leaf)
        else:
            # raise value error
            raise ValueError("min_samples_leaf must be an integer"
                             ", got %s" % min_samples_leaf)

        # check share of features in splitting
        if isinstance(max_features, float):
            # check if its within (0,1]
            if (max_features > 0 and max_features <= 1):
                # assign the input value
                self.max_features = max_features
            else:
                # raise value error
                raise ValueError("max_features must be within (0,1]"
                                 ", got %s" % max_features)
        else:
            # raise value error
            raise ValueError("max_features must be a float"
                             ", got %s" % max_features)

        # initialize orf
        self.forest = None
        # initialize performance metrics
        self.confusion = None
        self.measures = None

    # function to estimate ordered forest
    def fit(self, X, y, verbose=False):
        """
        Ordered Forest estimation.

        Parameters
        ----------
        X : TYPE: pd.DataFrame
            DESCRIPTION: matrix of covariates.
        y : TYPE: pd.Series
            DESCRIPTION: vector of outcomes.
        verbose : TYPE: bool
            DESCRIPTION: should be the results printed to console?
            Default is False.

        Returns
        -------
        result: ordered probability predictions by Ordered Forest.
        """
        # check if features X are a pandas dataframe
        self.__xcheck(X)

        # check if outcome y is a pandas series
        if isinstance(y, pd.Series):
            # check if its non-empty
            if y.empty:
                # raise value error
                raise ValueError("y Series is empty. Check the input.")
        else:
            # raise value error
            raise ValueError("y is not a Pandas Series. Recode the input.")

        # get the number of outcome classes
        nclass = len(y.unique())
        # define the labels if not supplied using list comprehension
        labels = ['Class ' + str(c_idx) for c_idx in range(1, nclass + 1)]
        # create an empty dictionary to save the forests
        forests = {}
        # create an empty dictionary to save the predictions
        probs = {}

        # estimate random forest on each class outcome except the last one
        for class_idx in range(1, nclass, 1):
            # create binary outcome indicator for the outcome in the forest
            outcome_ind = (y <= class_idx) * 1
            # call rf from scikit learn and save it in dictionary
            forests[class_idx] = RandomForestRegressor(
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                oob_score=True)
            # fit the model with the binary outcome
            forests[class_idx].fit(X=X, y=outcome_ind)
            # get the in-sample predictions, i.e. the out-of-bag predictions
            probs[class_idx] = pd.Series(forests[class_idx].oob_prediction_,
                                         name=labels[class_idx - 1],
                                         index=X.index)

        # collect predictions into a dataframe
        probs = pd.DataFrame(probs)
        # create 2 distinct matrices with zeros and ones for easy subtraction
        probs_0 = pd.concat([pd.Series(np.zeros(probs.shape[0]),
                                       index=probs.index,
                                       name=0), probs], axis=1)
        probs_1 = pd.concat([probs, pd.Series(np.ones(probs.shape[0]),
                                              index=probs.index, name=nclass)],
                            axis=1)

        # difference out the adjacent categories to singleout the class probs
        class_probs = probs_1 - probs_0.values
        # check if some probabilities become negative and set them to zero
        class_probs[class_probs < 0] = 0
        # normalize predictions to sum up to 1 after non-negativity correction
        class_probs = class_probs.divide(class_probs.sum(axis=1), axis=0)
        # set the new column names according to specified class labels
        class_probs.columns = labels

        # pack estimated forest and class predictions into output dictionary
        self.forest = {'forests': forests, 'probs': class_probs}
        # compute prediction performance
        self.__performance(y)
        # check if performance metrics should be printed
        if verbose:
            self.performance()

        # return the output
        return self

    # function to predict with estimated ordered forest
    def predict(self, X):
        """
        Ordered Forest prediction.

        Parameters
        ----------
        X : TYPE: pd.DataFrame
            DESCRIPTION: matrix of covariates.

        Returns
        -------
        result: ordered probability predictions by Ordered Forest.
        """
        # check if features X are a pandas dataframe
        self.__xcheck(X)

        # get the estimated forests
        forests = self.forest['forests']
        # get the class labels
        labels = list(self.forest['probs'].columns)
        # get the number of outcome classes
        nclass = len(labels)
        # create an empty dictionary to save the predictions
        probs = {}

        # estimate random forest on each class outcome except the last one
        for class_idx in range(1, nclass, 1):
            # predict with the estimated forests out-of-sample
            probs[class_idx] = pd.Series(forests[class_idx].predict(X=X),
                                         name=labels[class_idx - 1],
                                         index=X.index)

        # collect predictions into a dataframe
        probs = pd.DataFrame(probs)
        # create 2 distinct matrices with zeros and ones for easy subtraction
        probs_0 = pd.concat([pd.Series(np.zeros(probs.shape[0]),
                                       index=probs.index,
                                       name=0), probs], axis=1)
        probs_1 = pd.concat([probs, pd.Series(np.ones(probs.shape[0]),
                                              index=probs.index, name=nclass)],
                            axis=1)

        # difference out the adjacent categories to singleout the class probs
        class_probs = probs_1 - probs_0.values
        # check if some probabilities become negative and set them to zero
        class_probs[class_probs < 0] = 0
        # normalize predictions to sum up to 1 after non-negativity correction
        class_probs = class_probs.divide(class_probs.sum(axis=1), axis=0)
        # set the new column names according to specified class labels
        class_probs.columns = labels

        # return the class predictions
        return class_probs

    # function to evaluate marginal effects with estimated ordered forest
    def margin(self, X, verbose=False):
        """
        Ordered Forest prediction.

        Parameters
        ----------
        X : TYPE: pd.DataFrame
            DESCRIPTION: matrix of covariates.
        verbose : TYPE: bool
            DESCRIPTION: should be the results printed to console?
            Default is False.

        Returns
        -------
        result: Mean marginal effects by Ordered Forest.
        """
        # check if features X are a pandas dataframe
        self.__xcheck(X)

        # get the class labels
        labels = list(self.forest['probs'].columns)
        # define the window size share for evaluating the effect
        h_std = 0.1
        # create empty dataframe to store marginal effects
        margins = pd.DataFrame(index=X.columns, columns=labels)

        # loop over all covariates
        for x_id in list(X.columns):
            # first check if its dummy, categorical or continuous
            if list(np.sort(X[x_id].unique())) == [0, 1]:
                # compute the marginal effect as a discrete change in probs
                # save original values of the dummy variable
                dummy = np.array(X[x_id])
                # set x=1
                X[x_id] = 1
                prob_x1 = self.predict(X=X)
                # set x=0
                X[x_id] = 0
                prob_x0 = self.predict(X=X)
                # take the differences and columns means
                effect = (prob_x1 - prob_x0).mean(axis=0)
                # reset the dummy into the original values
                X[x_id] = dummy
            else:
                # compute the marginal effect as continuous change in probs
                # save original values of the continous variable
                original = np.array(X[x_id])
                # get the min and max of x for the support check
                x_min = original.min()
                x_max = original.max()
                # get the standard deviation of x for marginal effect
                x_std = original.std()
                # set x_up=x+h_std*x_std
                x_up = original + (h_std * x_std)
                # check if x_up is within the support of x
                x_up = ((x_up < x_max) * x_up + (x_up >= x_max) * x_max)
                x_up = ((x_up > x_min) * x_up + (x_up <= x_min) *
                        (x_min + h_std * x_std))
                # check if x is categorical and adjust to integers accordingly
                if len(X[x_id].unique()) <= 10:
                    # set x_up=ceiling(x_up)
                    x_up = np.ceil(x_up)
                # replace the x with x_up
                X[x_id] = x_up
                # get orf predictions
                prob_x1 = self.predict(X=X)
                # set x_down=x-h_std*x_std
                x_down = original - (h_std * x_std)
                # check if x_down is within the support of x
                x_down = ((x_down > x_min) * x_down + (x_down <= x_min) *
                          x_min)
                x_down = ((x_down < x_max) * x_down + (x_down >= x_max) *
                          (x_max - h_std * x_std))
                # check if x is categorical and adjust to integers accordingly
                if len(X[x_id].unique()) <= 10:
                    # set x_down=floor(x_down)
                    x_down = np.floor(x_down)
                # replace the x with x_down
                X[x_id] = x_down
                # get orf predictions
                prob_x0 = self.predict(X=X)
                # take the differences, scale them and take columns means
                diff = prob_x1 - prob_x0
                # define scaling parameter
                scale = pd.Series((x_up - x_down), index=X.index)
                # rescale the differences and take the column means
                effect = diff.divide(scale, axis=0).mean(axis=0)
                # reset x into the original values
                X[x_id] = original
            # assign the effects into the output dataframe
            margins.loc[x_id, :] = effect

        # redefine all effect results as floats
        margins = margins.astype(float)

        # check if marginal effects should be printed
        if verbose:
            # print marginal effects nicely
            print('Ordered Forest: Mean Marginal Effects', '-' * 80,
                  margins, '-' * 80, '\n\n', sep='\n')

        # return marginal effects
        return margins

    # performance measures (private method, not available to user)
    def __performance(self, y):
        """
        Evaluate the prediction performance using MSE and CA.

        Parameters
        ----------
        y : TYPE: pd.Series
            DESCRIPTION: vector of outcomes.

        Returns
        -------
        None. Calculates MSE, Classification accuracy and confusion matrix.
        """
        # take over needed values
        predictions = self.forest['probs']

        # compute the mse: version 1
        # create storage empty dataframe
        mse_matrix = pd.DataFrame(0, index=y.index,
                                  columns=predictions.columns)
        # allocate indicators for true outcome and leave zeros for the others
        # minus 1 for the column index as indices start with 0, outcomes with 1
        for obs_idx in range(len(y)):
            mse_matrix.iloc[obs_idx, y.iloc[obs_idx] - 1] = 1
        # compute mse directly now by substracting two dataframes and rowsums
        mse_1 = np.mean(((mse_matrix - predictions) ** 2).sum(axis=1))

        # compute the mse: version 2
        # create storage for modified predictions
        modified_pred = pd.Series(0, index=y.index)
        # modify the predictions with 1*P(1)+2*P(2)+3*P(3) as an alternative
        for class_idx in range(len(predictions.columns)):
            # add modified predictions together for all class values
            modified_pred = (modified_pred +
                             (class_idx + 1) * predictions.iloc[:, class_idx])
        # compute the mse directly now by substracting two series and mean
        mse_2 = np.mean((y - modified_pred) ** 2)

        # compute classification accuracy
        # define classes with highest probability (+1 as index starts with 0)
        class_pred = pd.Series((predictions.values.argmax(axis=1) + 1),
                               index=y.index)
        # the accuracy directly now by mean of matching classes
        acc = np.mean(y == class_pred)

        # create te confusion matrix
        self.confusion = pd.DataFrame(index=predictions.columns,
                                      columns=predictions.columns)
        # fill in the matrix by comparisons
        # loop over the actual outcomes
        for actual in range(len(self.confusion)):
            # loop over the predicted outcomes
            for predicted in range(len(self.confusion)):
                # compare the actual with predicted and sum it up
                self.confusion.iloc[actual, predicted] = sum(
                    (y == actual + 1) & (class_pred == predicted + 1))

        # wrap the results into a dataframe
        self.measures = pd.DataFrame({'mse 1': mse_1, 'mse 2': mse_2,
                                      'accuracy': acc}, index=['value'])

        # empty return
        return None

    # performance measures (public method, available to user)
    def performance(self):
        """
        Print the prediction performance based on MSE and CA.

        Parameters
        ----------
        None.

        Returns
        -------
        None. Prints MSE, Classification accuracy and confusion matrix.
        """
        # print the result
        print('Prediction Performance of Ordered Forest', '-' * 80,
              self.measures, '-' * 80, '\n\n', sep='\n')

        # print the confusion matrix
        print('Confusion Matrix for Ordered Forest', '-' * 80,
              '                         Predictions ', '-' * 80,
              self.confusion, '-' * 80, '\n\n', sep='\n')

        # empty return
        return None

    # check user input for covariates (private method, not available to user)
    def __xcheck(self, X):
        """
        Check the user input for the pandas dataframe of covariates.

        Parameters
        ----------
        X : TYPE: pd.DataFrame
            DESCRIPTION: matrix of covariates.

        Returns
        -------
        None. Checks for the correct user input.
        """
        # check if features X are a pandas dataframe
        if isinstance(X, pd.DataFrame):
            # check if its non-empty
            if X.empty:
                # raise value error
                raise ValueError("X DataFrame is empty. Check the input.")
        else:
            # raise value error
            raise ValueError("X is not a Pandas DataFrame. Recode the input.")

        # empty return
        return None
