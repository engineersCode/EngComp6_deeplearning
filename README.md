# Engineering Computations Module 6

_Engineering Computations_ is a collection of stackable learning modules, flexible for adoption in different situations.
It aims to develop computational skills for students in engineering, but it can also be used by students in other science majors.
The course uses the Python programming language and the Jupyter open-source tools for interactive computing.

## Module 6: deep learning

*A step-by-step introduction to deep learning (a.k.a. neural network) models for scientists and engineers*

**Pre-requisite: learning modules [*EngComp 1*](https://github.com/engineersCode/EngComp1_offtheground) and [*EngComp 4*](https://github.com/engineersCode/EngComp4_landlinear) of our collection.** Recommended: [*EngComp 2*](https://github.com/engineersCode/EngComp2_takeoff), or basic use of `pandas` for data manipulation.

### [Lesson 1](http://go.gwu.edu/engcomp6lesson1): Linear regression by gradient descent

Find the minimum of a function by gradient descent. Play with SymPy. Key ingredients of building a linear model from data with a single independent variable. Optimize a loss function to find the model parameters.

### [Lesson 2](http://go.gwu.edu/engcomp6lesson2): Logistic regression

Composition of a linear model with the logistic function. Construct the logistic loss function by integration. Find the model parameters with `autograd`. Combine with a decision boundary to do classification.

### [Lesson 3](http://go.gwu.edu/engcomp6lesson3): Multiple linear regression

Use multiple independent variables to build a linear model. Express multiple linear regression in matrix form. Find the weights by gradient descent. Scale (normalize) the features to ensure convergence. Get acquainted with `scikit-learn`. Model accuracy. Linear regression with `scikit-learn` and with pseudo-inverse.

### [Lesson 4](http://go.gwu.edu/engcomp6lesson4): Polynomial regression

Fitting a polynomial to data is a special case of multiple linear regression. Build polynomial features, scale the data, and train the model like in Lesson 3. For predictions with the model, use the scaling from the training data on the new data. 
Observe underfitting and overfitting. Use regularization to avoid overfitting. This is also called _ridge regression_. 
Do it with scikit-learn's `Ridge()`.

### [Lesson 5](http://go.gwu.edu/engcomp6lesson5): Multiple logistic regression (work-in-progress)

### [Lesson 6]() Neural network model (coming soon)

## Copyright and License

(c) 2021 Lorena A. Barba, Pi-Yueh Chuang, Tingyu Wang. 
All content is under Creative Commons Attribution [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode.txt), and all [code is under BSD-3 clause](https://github.com/engineersCode/EngComp/blob/master/LICENSE). We are happy if you re-use the content in any way!

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) [![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
