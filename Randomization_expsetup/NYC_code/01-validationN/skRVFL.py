# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 00:32:51 2020

@author: Javi
"""

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import numpy as np
from sklearn.linear_model import Ridge
from scipy.stats import mode

class RVFLClassifier (BaseEstimator, ClassifierMixin):

    """
    Random Vector Functional Link classifier
    """

    def __init__(self,
                 hid_num,
                 a=1):
        """
        Args:
        hid_num (int): number of hidden neurons
        a (int) : const value of sigmoid funcion
        """
        self.hid_num = hid_num
        self.a = a

    def _sigmoid(self, x):
        """
        sigmoid function
        Args:
        x (float): input

        Returns:
        float: output of sigmoid
        """
        sigmoid_range = 34.538776394910684
        x = np.clip(x, -sigmoid_range, sigmoid_range)
        return 1 / (1 + np.exp(-self.a * x))

    def _onehotencoder(self, n, label):
        """
        trasform label scalar to vector
        Args:
        n (int) : number of class, number of out layer neuron
        label (int) : label

        """
        return [-1 if i != label else 1 for i in range(1, n + 1)]

    def fit(self, X, y):
        """
        learning

        Args:
        X [[float]] array : feature vectors of learning data
        y [[float]] array : labels of leaning data
        """

        # number of class, number of output neuron
        self.out_num = max(y)

        if self.out_num != 1:
            y = np.array([self._onehotencoder(self.out_num, _y) for _y in y])

        # https://arxiv.org/pdf/1708.08282.pdf
        
        H1 = X.copy()

        # generate weights and bias between input layer and hidden layer
        
        np.random.seed()
        
        self.W = np.random.uniform(-1., 1.,(self.hid_num, X.shape[1]))
        self.bias = np.random.uniform(-1., 1.,(self.hid_num, 1))  
        
        H2 = self._sigmoid(np.dot(self.W, X.T)+self.bias).T
        
        HT = np.hstack((H1,H2))
        
        # find inverse weight matrix
        
        _H = np.linalg.pinv(HT)

        self.beta = np.dot(_H, y)
        
        return self

    def predict(self, X):
        
        """
        predict classify result

        Args:
        X [[float]] array: feature vectors of learning data

        Returns:
        [int]: labels of classification result
        """
        
        H1 = X.copy()
        H2 = self._sigmoid(np.dot(self.W, X.T)+self.bias).T
        HT = np.hstack((H1,H2))
        
        y = np.dot(HT, self.beta)

        if self.out_num == 1:
            return np.sign(y)
        else:
            return np.argmax(y, 1) + np.ones(y.shape[0])


class RVFLRegressor(BaseEstimator, RegressorMixin):

    """
    Random Vector Functional Link regressor
    """

    def __init__(self,
                 hid_num,
                 a=1,alphaRidge=1.0):
        """
        Args:
        hid_num (int): number of hidden neurons
        a (int) : const value of sigmoid funcion
        """
        self.hid_num = hid_num
        self.a = a
        self.alphaRidge = alphaRidge

    def _sigmoid(self, x):
        """
        sigmoid function
        Args:
        x (float): input

        Returns:
        float: output of sigmoid
        """
        sigmoid_range = 34.538776394910684
        x = np.clip(x, -sigmoid_range, sigmoid_range)
        return 1 / (1 + np.exp(-self.a * x))

    def fit(self, X, y):
        """
        learning

        Args:
        X [[float]] array : feature vectors of learning data
        y [[float]] array : target values of leaning data
        """

        # https://arxiv.org/pdf/1708.08282.pdf
        
        H1 = X.copy()

        # generate weights and bias between input layer and hidden layer
        
        np.random.seed()
        
        self.W = np.random.uniform(-1., 1.,(self.hid_num, X.shape[1]))
        self.bias = np.random.uniform(-1., 1.,(self.hid_num, 1))  
        
        H2 = self._sigmoid(np.dot(self.W, X.T)+self.bias).T
        
        HT = np.hstack((H1,H2))
        
        # find inverse weight matrix
        
        self.reg = Ridge(alpha=self.alphaRidge)
        self.reg.fit(HT,y)

    def predict(self, X):
        
        """
        predict classify result

        Args:
        X [[float]] array: feature vectors of learnig data

        Returns:
        [float]: predicted data
        """
        
        H1 = X.copy()
        H2 = self._sigmoid(np.dot(self.W, X.T) + self.bias).T
        HT = np.hstack((H1,H2))
        
        y = self.reg.predict(HT)

        return y
    
class multilayerRVFLClassifier(BaseEstimator, ClassifierMixin):

    """
    Random Vector Functional Link classifier
    """

    def __init__(self,
                 neuronsPerLayer,
                 a = 1):
        """
        Args:
        neuronsPerLayer [int]: number of hidden neurons per layer
        a (int) : const value of sigmoid funcion
        """
        self.neuronsPerLayer = neuronsPerLayer
        self.a = a

    def _sigmoid(self, x):
        """
        sigmoid function
        Args:
        x (float): input

        Returns:
        float: output of sigmoid
        """
        sigmoid_range = 34.538776394910684
        x = np.clip(x, -sigmoid_range, sigmoid_range)
        return 1 / (1 + np.exp(-self.a * x))

    def _onehotencoder(self, n, label):
        """
        trasform label scalar to vector
        Args:
        n (int) : number of class, number of out layer neuron
        label (int) : label
        """
        return [-1 if i != label else 1 for i in range(1, n + 1)]

    def fit(self, X, y):
        """
        learning

        Args:
        X [[float]] array : feature vectors of learning data
        y [[float]] array : labels of leaning data
        """
        # number of class, number of output neuron
        self.out_num = max(y)

        if self.out_num != 1:
            y = np.array([self._onehotencoder(self.out_num, _y) for _y in y])

        # https://arxiv.org/pdf/1708.08282.pdf
        
        np.random.seed()
        
        self.W = []
        self.bias = []
        
        H = [X.copy()]
        
        for hid_num in self.neuronsPerLayer:
            
            self.W.append(np.random.uniform(-1., 1.,(hid_num, H[-1].shape[1])))
            self.bias.append(np.random.uniform(-1., 1.,(hid_num, 1)))
            H2 = self._sigmoid(np.dot(self.W[-1], H[-1].T) + self.bias[-1]).T
            H.append(H2)

        HT = np.hstack((X.copy(),H[-1]))
        
        # find inverse weight matrix
        
        _H = np.linalg.pinv(HT)

        self.beta = np.dot(_H, y)
        
        return self

    def predict(self, X):
        
        """
        predict classify result

        Args:
        X [[float]] array: feature vectors of learnig data

        Returns:
        [int]: labels of classification result
        """
        
        H = [X.copy()]
        
        indexLayer = 0
        
        for hid_num in self.neuronsPerLayer:
            
            H2 = self._sigmoid(np.dot(self.W[indexLayer], H[-1].T) + self.bias[indexLayer]).T
            H.append(H2)
            indexLayer = indexLayer + 1
        
        HT = np.hstack((X.copy(),H[-1]))
        
        y = np.dot(HT, self.beta)

        if self.out_num == 1:
            return np.sign(y)
        else:
            return np.argmax(y, 1) + np.ones(y.shape[0])


class multilayerRVFLRegressor(BaseEstimator, RegressorMixin):

    """
    Random Vector Functional Link regressor
    """

    def __init__(self,
                 neuronsPerLayer,
                 a=1,alphaRidge=1.0):
        """
        Args:
        hid_num (int): number of hidden neurons per layer
        a (int) : const value of sigmoid funcion
        """
        self.neuronsPerLayer = neuronsPerLayer
        self.a = a
        self.alphaRidge = alphaRidge

    def _sigmoid(self, x):
        """
        sigmoid function
        Args:
        x (float): input

        Returns:
        float: output of sigmoid
        """
        sigmoid_range = 34.538776394910684
        x = np.clip(x, -sigmoid_range, sigmoid_range)
        return 1 / (1 + np.exp(-self.a * x))

    def fit(self, X, y):
        """
        learning

        Args:
        X [[float]] array : feature vectors of learning data
        y [[float]] array : target values of leaning data
        """

        # https://arxiv.org/pdf/1708.08282.pdf
        
        np.random.seed()
        
        self.W = []
        self.bias = []
        
        H = [X.copy()]
        
        for hid_num in self.neuronsPerLayer:
            
            self.W.append(np.random.uniform(-1., 1.,(hid_num, H[-1].shape[1])))
            self.bias.append(np.random.uniform(-1., 1.,(hid_num, 1)))
            H2 = self._sigmoid(np.dot(self.W[-1], H[-1].T) + self.bias[-1]).T
            H.append(H2)

        HT = np.hstack((X.copy(),H[-1]))
        
        
        # find inverse weight matrix
        
        self.reg = Ridge(alpha=self.alphaRidge)
        self.reg.fit(HT,y)

    def predict(self, X):
        
        """
        predict classify result

        Args:
        X [[float]] array: feature vectors of learnig data

        Returns:
        [float]: predicted values
        """
        
        H = [X.copy()]
        
        indexLayer = 0
        
        for hid_num in self.neuronsPerLayer:
            
            H2 = self._sigmoid(np.dot(self.W[indexLayer], H[-1].T) + self.bias[indexLayer]).T
            H.append(H2)
            indexLayer = indexLayer + 1
        
        HT = np.hstack((X.copy(),H[-1]))
        
        y = self.reg.predict(HT)

        return y
    
class multilayerELMRegressor(BaseEstimator, RegressorMixin):

    """
    multilayer Extreme Learning Machine regressor
    """

    def __init__(self,
                 neuronsPerLayer,
                 a=1,alphaRidge=1.0):
        """
        Args:
        hid_num (int): number of hidden neurons per layer
        a (int) : const value of sigmoid funcion
        """
        self.neuronsPerLayer = neuronsPerLayer
        self.a = a
        self.alphaRidge = alphaRidge

    def _sigmoid(self, x):
        """
        sigmoid function
        Args:
        x (float): input

        Returns:
        float: output of sigmoid
        """
        sigmoid_range = 34.538776394910684
        x = np.clip(x, -sigmoid_range, sigmoid_range)
        return 1 / (1 + np.exp(-self.a * x))

    def fit(self, X, y):
        """
        learning

        Args:
        X [[float]] array : feature vectors of learning data
        y [[float]] array : target values of leaning data
        """

        # https://arxiv.org/pdf/1708.08282.pdf
        
        np.random.seed()
        
        self.W = []
        # self.bias = []
        
        H = [X.copy()]
        
        for hid_num in self.neuronsPerLayer:
            
            self.W.append(np.random.uniform(-1., 1.,(hid_num, H[-1].shape[1])))
            # self.bias.append(np.random.uniform(-1., 1.,(hid_num, 1)))
            H2 = self._sigmoid(np.dot(self.W[-1], H[-1].T)).T
            H.append(H2)

        HT = H[-1].copy()
        
        
        # find inverse weight matrix
        
        self.reg = Ridge(alpha=self.alphaRidge)
        self.reg.fit(HT,y)

    def predict(self, X):
        
        """
        predict classify result

        Args:
        X [[float]] array: feature vectors of learnig data

        Returns:
        [float]: predicted values
        """
        
        H = [X.copy()]
        
        indexLayer = 0
        
        for hid_num in self.neuronsPerLayer:
            
            H2 = self._sigmoid(np.dot(self.W[indexLayer], H[-1].T)).T
            H.append(H2)
            indexLayer = indexLayer + 1
        
        HT = H[-1].copy()
        
        y = self.reg.predict(HT)

        return y
    
    
class deepRVFLClassifier(BaseEstimator, ClassifierMixin):

    """
    Random Vector Functional Link classifier
    """

    def __init__(self,
                 neuronsPerLayer,
                 a = 1):
        """
        Args:
        neuronsPerLayer [int]: number of hidden neurons per layer
        a (int) : const value of sigmoid funcion
        """
        self.neuronsPerLayer = neuronsPerLayer
        self.a = a

    def _sigmoid(self, x):
        """
        sigmoid function
        Args:
        x (float): input

        Returns:
        float: output of sigmoid
        """
        sigmoid_range = 34.538776394910684
        x = np.clip(x, -sigmoid_range, sigmoid_range)
        return 1 / (1 + np.exp(-self.a * x))

    def _onehotencoder(self, n, label):
        """
        trasform label scalar to vector
        Args:
        n (int) : number of class, number of out layer neuron
        label (int) : label
        """
        return [-1 if i != label else 1 for i in range(1, n + 1)]

    def fit(self, X, y):
        """
        learning

        Args:
        X [[float]] array : feature vectors of learning data
        y [[float]] array : labels of leaning data
        """
        # number of class, number of output neuron
        self.out_num = max(y)

        if self.out_num != 1:
            y = np.array([self._onehotencoder(self.out_num, _y) for _y in y])

        # https://arxiv.org/pdf/1708.08282.pdf
        
        np.random.seed()
        
        self.W = []
        self.bias = []
        
        H = [X.copy()]
        
        for hid_num in self.neuronsPerLayer:
            
            self.W.append(np.random.uniform(-1., 1.,(hid_num, H[-1].shape[1])))
            self.bias.append(np.random.uniform(-1., 1.,(hid_num, 1)))
            H2 = self._sigmoid(np.dot(self.W[-1], H[-1].T) + self.bias[-1]).T
            H.append(H2)
        

        HT = np.hstack(H)
        
        # find inverse weight matrix
        
        _H = np.linalg.pinv(HT)

        self.beta = np.dot(_H, y)
        
        return self

    def predict(self, X):
        
        """
        predict classify result

        Args:
        X [[float]] array: feature vectors of learnig data

        Returns:
        [int]: labels of classification result
        """
        
        H = [X.copy()]
        
        indexLayer = 0
        
        for hid_num in self.neuronsPerLayer:
            
            H2 = self._sigmoid(np.dot(self.W[indexLayer], H[-1].T) + self.bias[indexLayer]).T
            H.append(H2)
            indexLayer = indexLayer + 1
        
        HT = np.hstack(H)
        
        y = np.dot(HT, self.beta)

        if self.out_num == 1:
            return np.sign(y)
        else:
            return np.argmax(y, 1) + np.ones(y.shape[0])


class deepRVFLRegressor(BaseEstimator, RegressorMixin):

    """
    Random Vector Functional Link regressor
    """

    def __init__(self,
                 neuronsPerLayer,
                 a=1,alphaRidge=1.0):
        """
        Args:
        hid_num (int): number of hidden neurons per layer
        a (int) : const value of sigmoid funcion
        """
        self.neuronsPerLayer = neuronsPerLayer
        self.a = a
        self.alphaRidge = alphaRidge

    def _sigmoid(self, x):
        """
        sigmoid function
        Args:
        x (float): input

        Returns:
        float: output of sigmoid
        """
        sigmoid_range = 34.538776394910684
        x = np.clip(x, -sigmoid_range, sigmoid_range)
        return 1 / (1 + np.exp(-self.a * x))

    def fit(self, X, y):
        """
        learning

        Args:
        X [[float]] array : feature vectors of learning data
        y [[float]] array : target values of leaning data
        """

        # https://arxiv.org/pdf/1708.08282.pdf
        
        np.random.seed()
        
        self.W = []
        self.bias = []
        
        H = [X.copy()]
        
        for hid_num in self.neuronsPerLayer:
            
            self.W.append(np.random.uniform(-1., 1.,(hid_num, H[-1].shape[1])))
            self.bias.append(np.random.uniform(-1., 1.,(hid_num, 1)))
            H2 = self._sigmoid(np.dot(self.W[-1], H[-1].T) + self.bias[-1]).T
            H.append(H2)

        HT = np.hstack(H)
        
        # find inverse weight matrix
        
        self.reg = Ridge(alpha=self.alphaRidge)
        self.reg.fit(HT,y)

    def predict(self, X):
        
        """
        predict classify result

        Args:
        X [[float]] array: feature vectors of learnig data

        Returns:
        [float]: predicted values
        """
        
        H = [X.copy()]
        
        indexLayer = 0
        
        for hid_num in self.neuronsPerLayer:
            
            H2 = self._sigmoid(np.dot(self.W[indexLayer], H[-1].T) + self.bias[indexLayer]).T
            H.append(H2)
            indexLayer = indexLayer + 1
        
        HT = np.hstack(H)
        
        y = self.reg.predict(HT)

        return y
    
class ensembleDeepRVFLClassifier(BaseEstimator, ClassifierMixin):

    """
    Random Vector Functional Link classifier
    """

    def __init__(self,
                 neuronsPerLayer,
                 a = 1):
        """
        Args:
        neuronsPerLayer [int]: number of hidden neurons per layer
        a (int) : const value of sigmoid funcion
        """
        self.neuronsPerLayer = neuronsPerLayer
        self.a = a

    def _sigmoid(self, x):
        """
        sigmoid function
        Args:
        x (float): input

        Returns:
        float: output of sigmoid
        """
        sigmoid_range = 34.538776394910684
        x = np.clip(x, -sigmoid_range, sigmoid_range)
        return 1 / (1 + np.exp(-self.a * x))

    def _onehotencoder(self, n, label):
        """
        trasform label scalar to vector
        Args:
        n (int) : number of class, number of out layer neuron
        label (int) : label

        """
        return [-1 if i != label else 1 for i in range(1, n + 1)]

    def fit(self, X, y):
        """
        learning

        Args:
        X [[float]] array : feature vectors of learning data
        y [[float]] array : labels of leaning data
        """
        # number of class, number of output neuron
        self.out_num = max(y)

        if self.out_num != 1:
            y = np.array([self._onehotencoder(self.out_num, _y) for _y in y])

        # https://arxiv.org/pdf/1907.00350.pdf
        
        np.random.seed()
        
        self.W = []
        self.bias = []
        self.betas = []
        
        H = [X.copy()]
        
        for hid_num in self.neuronsPerLayer:
            
            self.W.append(np.random.uniform(-1., 1.,(hid_num, H[-1].shape[1])))
            self.bias.append(np.random.uniform(-1., 1.,(hid_num, 1)))
            H2 = self._sigmoid(np.dot(self.W[-1], H[-1].T) + self.bias[-1]).T
            H.append(np.hstack((X,H2)))
            _H = np.linalg.pinv(H[-1])
            self.betas.append(np.dot(_H, y))
        
        return self

    def predict(self, X):
        
        """
        predict classify result

        Args:
        X [[float]] array: feature vectors of learnig data

        Returns:
        [int]: labels of classification result
        """
        
        H = [X.copy()]
        indexLayer = 0
        
        self.ypred = []
        
        for hid_num in self.neuronsPerLayer:
            
            H2 = self._sigmoid(np.dot(self.W[indexLayer], H[-1].T) + self.bias[indexLayer]).T
            H.append(np.hstack((X,H2)))
            y = np.dot(H[-1], self.betas[indexLayer])
            if self.out_num == 1:
                self.ypred.append(np.sign(y))
            else:
                self.ypred.append(np.argmax(y, 1) + np.ones(y.shape[0]))
            
            indexLayer = indexLayer + 1
        
        return mode(self.ypred,0)[0][0]


class ensembleDeepRVFLRegressor(BaseEstimator, RegressorMixin):

    """
    Random Vector Functional Link regressor
    """

    def __init__(self,
                 neuronsPerLayer,
                 a=1,alphaRidge=1.0):
        """
        Args:
        hid_num (int): number of hidden neurons
        a (int) : const value of sigmoid funcion
        """
        self.neuronsPerLayer = neuronsPerLayer
        self.a = a
        self.alphaRidge = alphaRidge

    def _sigmoid(self, x):
        """
        sigmoid function
        Args:
        x (float): input

        Returns:
        float: output of sigmoid
        """
        sigmoid_range = 34.538776394910684
        x = np.clip(x, -sigmoid_range, sigmoid_range)
        return 1 / (1 + np.exp(-self.a * x))

    def fit(self, X, y):
        """
        learning

        Args:
        X [[float]] array : feature vectors of learning data
        y [[float]] array : target values of leaning data
        """

        # https://arxiv.org/pdf/1907.00350.pdf
        
        np.random.seed()
        
        self.W = []
        self.bias = []
        self.reg = []
        
        H = [X.copy()]
        
        for hid_num in self.neuronsPerLayer:
            
            self.W.append(np.random.uniform(-1., 1.,(hid_num, H[-1].shape[1])))
            self.bias.append(np.random.uniform(-1., 1.,(hid_num, 1)))
            H2 = self._sigmoid(np.dot(self.W[-1], H[-1].T) + self.bias[-1]).T
            H.append(np.hstack((X,H2)))
            self.reg.append(Ridge(alpha=self.alphaRidge))
            self.reg[-1].fit(H[-1],y)
        
        return self

    def predict(self, X):
        
        """
        predict classify result

        Args:
        X [[float]] array: feature vectors of learnig data

        Returns:
        [float]: predicted values
        """
        
        H = [X.copy()]
        indexLayer = 0
        
        self.ypred = []
        
        for hid_num in self.neuronsPerLayer:
            
            H2 = self._sigmoid(np.dot(self.W[indexLayer], H[-1].T) + self.bias[indexLayer]).T
            H.append(np.hstack((X,H2)))
            y = self.reg[indexLayer].predict(H[-1])
            self.ypred.append(y)
            indexLayer = indexLayer + 1
        
        return np.mean(self.ypred,0)
    

# =============================================================================
# SHOWTIME!
# =============================================================================

# from sklearn import preprocessing
# from sklearn.model_selection import KFold, cross_val_score

# hid_nums = 10 # only for RVFL
# neuronsPerLayer = (50,50,50,50,50,50,50,50) # for deep variants

# print('**********************************************************')
# print('Classification')
# print('**********************************************************')

# from sklearn.datasets import load_digits
# X, y = load_digits(return_X_y=True)

# X = preprocessing.StandardScaler().fit_transform(X)

# # =============================================================================

# print('RVFL Classifier:')
# e = RVFLClassifier(hid_num=hid_nums)

# ave = 0

# for i in range(10):
#     cv = KFold(n_splits=5, shuffle=True)
#     scores = cross_val_score(e, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
#     ave += scores.mean()

# ave /= 10
# print("Accuracy: %0.3f " % (ave))

# # =============================================================================

# print('Multi-layer RVFL Classifier:')
# e = multilayerRVFLClassifier(neuronsPerLayer=neuronsPerLayer)

# ave = 0

# for i in range(10):
#     cv = KFold(n_splits=5, shuffle=True)
#     scores = cross_val_score(e, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
#     ave += scores.mean()

# ave /= 10
# print("Accuracy: %0.3f " % (ave))

# # =============================================================================

# print('Deep RVFL Classifier:')
# e = deepRVFLClassifier(neuronsPerLayer=neuronsPerLayer)

# ave = 0

# for i in range(10):
#     cv = KFold(n_splits=5, shuffle=True)
#     scores = cross_val_score(e, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
#     ave += scores.mean()

# ave /= 10
# print("Accuracy: %0.3f " % (ave))

# # =============================================================================

# print('Ensemble Deep RVFL Classifier:')
# e = ensembleDeepRVFLClassifier(neuronsPerLayer=neuronsPerLayer)

# ave = 0

# for i in range(10):
#     cv = KFold(n_splits=5, shuffle=True)
#     scores = cross_val_score(e, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
#     ave += scores.mean()

# ave /= 10
# print("Accuracy: %0.3f " % (ave))

# print('**********************************************************')
# print('Regression')
# print('**********************************************************')

# from sklearn.datasets import load_boston
# X, y = load_boston(return_X_y=True)

# X = preprocessing.StandardScaler().fit_transform(X)
# y = np.ravel(preprocessing.StandardScaler().fit_transform(y.reshape(-1,1)))

# # =============================================================================

# print('RVFL Regressor:')
# e = RVFLRegressor(hid_num=hid_nums)

# ave = 0

# for i in range(10):
#     cv = KFold(n_splits=5, shuffle=True)
#     scores = cross_val_score(e, X, y, cv=cv, scoring='r2', n_jobs=-1)
#     ave += scores.mean()

# ave /= 10
# print("R2: %0.3f " % (ave))

# # =============================================================================

# print('Multi-layer RVFL Regressor:')
# e = multilayerRVFLRegressor(neuronsPerLayer=neuronsPerLayer)

# ave = 0

# for i in range(10):
#     cv = KFold(n_splits=5, shuffle=True)
#     scores = cross_val_score(e, X, y, cv=cv, scoring='r2', n_jobs=-1)
#     ave += scores.mean()

# ave /= 10
# print("R2: %0.3f " % (ave))

# # =============================================================================

# print('Deep RVFL Regressor:')
# e = deepRVFLRegressor(neuronsPerLayer=neuronsPerLayer)

# ave = 0

# for i in range(10):
#     cv = KFold(n_splits=5, shuffle=True)
#     scores = cross_val_score(e, X, y, cv=cv, scoring='r2', n_jobs=-1)
#     ave += scores.mean()

# ave /= 10
# print("R2: %0.3f " % (ave))

# # =============================================================================

# print('Ensemble Deep RVFL Regressor:')
# e = ensembleDeepRVFLRegressor(neuronsPerLayer=neuronsPerLayer)

# ave = 0

# for i in range(10):
#     cv = KFold(n_splits=5, shuffle=True)
#     scores = cross_val_score(e, X, y, cv=cv, scoring='r2', n_jobs=-1)
#     ave += scores.mean()

# ave /= 10
# print("R2: %0.3f " % (ave))

