# ml-package
###Own implementations of four machine learning algorithms using NumPy

Current algorithms include:
1) Linear Regression

    i) Standard least squares regression
    
    ii) Ridge regression
2) Logistic Regression (binary)
    
    i) Standard solver
    
    ii) Newton-method solver (2nd degree approximation)
    
3) K-Nearest Neighbors (supports multi-class)
4) Naive Bayes (supports multi-class)


To complement these algorithms, this package also 
contains methods performing functions such as:
1) Train-Test splits
2) Cross validation
3) Test set prediction
4) Confusion matrix calculator
5) Accuracy score calculator
6) RMSE score calculator
7) Polynomial feature expansions

Please note that there is minimal input checking, so please
review source code to diagnose sources of error.