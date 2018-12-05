# Project

This project is related to a cooperative bank specializing in the healthcare market.

The task is to create the best possible forecasting model for a marketing campaign for a new product "MedTrust" and to predict the completion probabilities.

Create various predictive models (logistic regression, decision trees, and random forests).

# Installation

### Download the Data

Download the data files from FarnoushSh into the data directory.
* You can find the train and test datasets [here](https://github.com/FarnoushSh/for-test/tree/master/Data).
* You'll need to register with FarnoushSh to download the data.

### Install the requirements
* Install the requirements using `pip install -r requirements.txt`.
    *Make sure you use Python 3.
    *You may want to use a virtual environment for this.

# Usage
* Run python DataLab.py.
    * This loads the train and test datasetes.
    * Makes some plots to visualize the data to find the outliers and missing values
        *plots give idea about the features related to the target variable
    *removes the outliers and missing values
    *balances the datasets
    *runs the Machine learning algorithms
        *prints the accuracy score.
        *plots AUC curves and box plots for comparing
    *prints the prediction of the probability of completion of buying.
