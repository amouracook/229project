# 229project

**Abstract:**

The field of post-disaster recovery modeling is currently limited by semi-heuristic assumptions. The ability to establish robust relationships is hindered by categorical, survey-based data. This project explores the use of regression methods, decision tree models, and neural networks to predict a household's post-disaster location of temporary shelter, a decision with significant implications on population out-migration. The resulting models perform substantially better than a random-guess trivial classifier but illustrate the challenges of class-imbalanced data.

**Data:**

Public Use Microdata Sample from the 2017 American Housing Survey, performed by the U.S. Census Bureau <br />
SF_41860.csv == data from San Francisco, Oakland, and Hayward, CA <br />
SJ_41940.csv == data from San Jose, Sunnyvalem and Santa Clara, CA <br />
CA_41860_31080.csv == data from San Francisco, Oakland, Hayward, Los Angeles, Long Beach, and Anaheim, CA <br />

**Code:**

feature_extraction.py == data cleaning function, options for one-hot encoding, SMOTE, and stratified train/test splitting<br />
PCA.py                == principal component analysis and plotting the data on axes for the first two principal components <br />
supervised.py         == implementation of logistic regression, ridge regression, random forest, and XGBoost supervised learning methods<br />
NN_CE.py              == implementation of a neural network using cross-entropy loss<br />
NN_WCE.py             == implementation of a neural network using weighted cross-entropy loss<br />
NN_BalCE.py           == implementation of a neural network using class-balanced cross-entropy loss<br />
NN_Focal.py           == implementation of a neural network using Focal Loss<br />

Other Folders:<br />
_ss        == superseded data, scripts, and files<br />
Figures    == ROC figures generated for the final report<br />
References == information about the American Housey Survey data
