#%%
#charging modules
#----------------------------------------------------------------------------------------------------------------------------#
# https://scikit-learn.org/stable/tutorial/basic/tutorial.html#machine-learning-the-problem-setting
from utils import dirpath, locateFileExt, locateFileNameExt
import os , sys, time, PIL
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

basepath_ = os.path.join(dirpath(),'canaux')
basepath_ = os.path.join(dirpath(),'previous_exp')

basepaths = locateFileNameExt(path=basepath_, containInName="stackImagePower", ext=".npy")
xlsxpaths = locateFileNameExt(path=basepath_, containInName="FlatSpectrum", ext=".xlsx")
basepaths.sort(); xlsxpaths.sort() 
dataColumns = pd.read_excel(xlsxpaths[1]).columns
#----------------------------------------------------------------------------------------------------------------------------#
Basepaths = [basepath.split('\\')[-1].split('.npy')[-2] for basepath in basepaths]
Xlsxpaths = [xlsxpath.split('\\')[-1].split('.xlsx')[-2] for xlsxpath in xlsxpaths]
print(list(zip(Basepaths,Xlsxpaths)))
# lstBasepaths = [basepath.split('\\')[-1] for basepath in basepaths]
#----------------------------------------------------------------------------------------------------------------------------#
#%%
A = {
        '01': (242, 233), '02': (274, 233), '03': (305, 233), '04': (336, 233), '05': (367, 233),
        '11': (242, 262), '12': (274, 262), '13': (305, 262), '14': (336, 262), '15': (367, 262),
        '21': (242, 294), '22': (274, 294), '23': (305, 294), '24': (336, 294), '25': (367, 294),
        '31': (242, 324), '32': (274, 324), '33': (305, 324), '34': (336, 324), '35': (367, 324),
        '41': (242, 354), '42': (274, 354), '43': (305, 354), '44': (336, 354), '45': (367, 354),
        '51': (242, 383), '52': (274, 383), '53': (305, 383), '54': (336, 383), '55': (367, 383),
        '61': (242, 412), '62': (274, 412), '63': (305, 412), '64': (336, 412), '65': (367, 412),
    }
Z = {   
        '_01': (242, 233), '_02': (274, 233), '_03': (305, 233), '_04': (336, 233), '_05': (367, 233),
        '_11': (242, 262), '_12': (274, 262), '_13': (305, 262), '_14': (336, 262), '_15': (367, 262),
        '_21': (242, 294), '_22': (274, 294),  '1': (305, 294),  '2': (336, 294),  '3': (367, 294),
        '13': (242, 324), '11': (274, 324),  '4': (305, 324),  '7': (336, 324),  '9': (367, 324),
        '14': (242, 354), '12': (274, 354),  '5': (305, 354),  '8': (336, 354), '10': (367, 354),
        '15': (242, 383), '16': (274, 383),  '6': (305, 383), '17': (336, 383), '18': (367, 383),
        '_61': (242, 412), '_62': (274, 412), '_63': (305, 412), '_64': (336, 412), '_65': (367, 412),
    } 
#%%
# Loading images Datas
#----------------------------------------------------------------------------------------------------------------------------#
[
stackImagePower2.5umP1C2P2C3_0_1,   data2_5umP1C2_500ms       = np.load(basepaths[0]),pd.read_excel(xlsxpaths[0]).to_numpy()[:,1:] 
stackImagePower2.5umP1C2_0_1,       data2_5umP1C2P2C3_500ms   = np.load(basepaths[1]),pd.read_excel(xlsxpaths[1]).to_numpy()[:,1:]
stackImagePower2.5umP1C3P2C2_0_1,   data2_5umP1C3P2C2_500ms   = np.load(basepaths[2]),pd.read_excel(xlsxpaths[2]).to_numpy()[:,1:]
stackImagePower2.5umP2C2_0_1,       data2_5umP2C2_500ms       = np.load(basepaths[3]),pd.read_excel(xlsxpaths[3]).to_numpy()[:,1:]
stackImagePower5.0umP1C2_0_1,       data5_0umP1C2_500ms       = np.load(basepaths[4]),pd.read_excel(xlsxpaths[4]).to_numpy()[:,1:]
stackImagepower5.0umP2C2_0_1,       data5_0umP2C2_500ms       = np.load(basepaths[5]),pd.read_excel(xlsxpaths[5]).to_numpy()[:,1:]

[('stackImagePower2.5umP1C2P2C3_0.1', 'FlatSpectrum2.5umP1C2'), ('stackImagePower2.5umP1C2_0.1', 'FlatSpectrum2.5umP1C2P2C3'), ('stackImagePower2.5umP1C3P2C2_0.1', 'FlatSpectrum2.5umP1C3P2C2'), ('stackImagePower2.5umP2C2_0.1', 'FlatSpectrum2.5umP2C2'), ('stackImagePower5.0umP1C2_0.1', 'FlatSpectrum5.0umP1C2')]

stackImageP1C2P2C3_0_1nm_1000ms,   dataP1C2P2C3_1000ms = np.load(basepaths[0]),np.delete(arr=pd.read_excel(xlsxpaths[1]).to_numpy()[:,1:], obj=(813), axis=0)
stackImageP1C2_0_1nm_1000ms,           dataP1C2_1000ms = np.load(basepaths[1]),pd.read_excel(xlsxpaths[0]).to_numpy()[:,1:]
stackImageP1C3P2C2_0_1nm_1000ms,   dataP1C3P2C2_1000ms = np.load(basepaths[2]),pd.read_excel(xlsxpaths[3]).to_numpy()[:,1:]
stackImageP1C3_0_1nm_1000ms,           dataP1C3_1000ms = np.load(basepaths[3]),np.delete(arr=pd.read_excel(xlsxpaths[2]).to_numpy()[:,1:], obj=(2,1084,1124), axis=0)    
stackImageP2C2_0_1nm_1000ms,           dataP2C2_1000ms = np.load(basepaths[4]),pd.read_excel(xlsxpaths[4]).to_numpy()[:,1:]
stackImageP2C3_0_1nm_1000ms,           dataP2C3_1000ms = np.load(basepaths[5]),np.delete(arr=pd.read_excel(xlsxpaths[5]).to_numpy()[:,1:], obj=(441), axis=0)
#----------------------------------------------------------------------------------------------------------------------------#
stackImageP1C2P2C3_0_1nm_1000ms,   dataP1C2P2C3_1000ms = np.load(basepaths[0]),np.delete(arr=pd.read_excel(xlsxpaths[1]).to_numpy()[:,1:], obj=(813), axis=0)
stackImageP1C2_0_1nm_1000ms,           dataP1C2_1000ms = np.load(basepaths[1]),pd.read_excel(xlsxpaths[0]).to_numpy()[:,1:]
stackImageP1C3P2C2_0_1nm_1000ms,   dataP1C3P2C2_1000ms = np.load(basepaths[2]),pd.read_excel(xlsxpaths[3]).to_numpy()[:,1:]
stackImageP1C3_0_1nm_1000ms,           dataP1C3_1000ms = np.load(basepaths[3]),np.delete(arr=pd.read_excel(xlsxpaths[2]).to_numpy()[:,1:], obj=(2,1084,1124), axis=0)    
stackImageP2C2_0_1nm_1000ms,           dataP2C2_1000ms = np.load(basepaths[4]),pd.read_excel(xlsxpaths[4]).to_numpy()[:,1:]
stackImageP2C3_0_1nm_1000ms,           dataP2C3_1000ms = np.load(basepaths[5]),np.delete(arr=pd.read_excel(xlsxpaths[5]).to_numpy()[:,1:], obj=(441), axis=0)

stackImagesDict = {
        "P1C2P2C3": (stackImageP1C2P2C3_0_1nm_1000ms,dataP1C2P2C3_1000ms),
        "P1C2":     (stackImageP1C2_0_1nm_1000ms,        dataP1C2_1000ms),
        "P1C3P2C2": (stackImageP1C3P2C2_0_1nm_1000ms,dataP1C3P2C2_1000ms),
        "P1C3":     (stackImageP1C3_0_1nm_1000ms,        dataP1C3_1000ms),
        "P2C2":     (stackImageP2C2_0_1nm_1000ms,        dataP2C2_1000ms),
        "P2C3":     (stackImageP2C3_0_1nm_1000ms,        dataP2C3_1000ms),
    }

#%%
# Data choice
#----------------------------------------------------------------------------------------------------------------------------#
stackImages, data = stackImagesDict['P1C2']
# Zimages = {f"{list(Z.keys())[i]}":image for i, image in enumerate(stackImages)}
Aimages = {f"{list(A.keys())[i]}":image for i, image in enumerate(stackImages)}

#select nanostructured Area
images = Aimages['31']

# #----------------------------------------------------------------------------------------------------------------------------#
img_ = images.reshape(images.shape[0],int(np.sqrt(images.shape[1])),int(np.sqrt(images.shape[1])))
img = img_[0,:,:]; plt.imshow(img, cmap='gray')
# #----------------------------------------------------------------------------------------------------------------------------#
#%
# Dataframe for vizualization
# https://pandas.pydata.org/docs/user_guide/merging.html
# displaying the DataFrame
from IPython.display import display

gain = (1.0, 1.2, 1.5, 2.0, 4.0, 6.0); gain = gain[1]*np.ones_like(data[:,:1])
intergrationTime = (100, 500, 1000); intergrationTime = intergrationTime[2]*np.ones_like(data[:,:1])
frame = [pd.DataFrame(data,columns=dataColumns[1:]), pd.DataFrame(intergrationTime,columns=['intergrationTime']), pd.DataFrame(gain,columns=['gain']), pd.DataFrame(images)]

datasFrame = pd.concat(frame, axis=1, join="inner"); datas = datasFrame.to_numpy()
display(datasFrame)

# #Model X_images -> lambdas, power, intergrationTime, gain
Xr_data, yr_data = images, datasFrame[['AOTF_input_wavelength(nm)','Power_Measured(W)','gain','intergrationTime']].to_numpy()

#%%
# Mixing data via cross validation : https://www.youtube.com/watch?v=w_bLGK4Pteo&t=39s
# https://scikit-learn.org/stable/modules/cross_validation.html
# #----------------------------------------------------------------------------------------------------------------------------#

from sklearn.datasets import make_regression
# to create a factice problem of regression
# create datasets
# #----------------------------------------------------------------------------------------------------------------------------#
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xr_data, yr_data, test_size=0.15, random_state=None, shuffle=True, stratify=None)

#%%
# Model Choice
# #----------------------------------------------------------------------------------------------------------------------------#
# https://scikit-learn.org/stable/modules/linear_model.html#generalized-linear-regression

#Choosing the model of regression
# from sklearn.linear_model import Lasso             ;  model = Lasso(alpha=0.1) ##Y=f(X)=f(X1,X2,..,Xn)
from sklearn.linear_model import LinearRegression  ;  model = LinearRegression(
                        fit_intercept=True,
                        normalize='deprecated',
                        copy_X=True, n_jobs=None,
                        positive=False
                        ) ##𝑓(𝐱) = 𝑏₀ + 𝑏₁𝑥₁ + ⋯ + 𝑏ᵣ𝑥ᵣ.
alpha = (0.5, 1.0)
from sklearn.linear_model import Ridge             ;  model = Ridge(
                        alpha=alpha[0], 
                        # fit_intercept=True,
                        # normalize='deprecated',
                        # copy_X=True, max_iter=None, 
                        # tol=0.001, solver='auto', 
                        # positive=False,
                        # random_state=None,
                        )
from sklearn.neighbors import KNeighborsRegressor  ;  model = KNeighborsRegressor(
                        n_neighbors=30,
                        # weights='uniform',
                        # algorithm='auto',
                        # leaf_size=30, p=2, 
                        metric='l2',
                        metric_params=None,
                        # n_jobs=None,
                        )

# https://machinelearningmastery.com/deep-learning-models-for-multi-output-regression/
# https://www.datatechnotes.com/2019/12/multi-output-regression-example-with.html
# https://machinelearningmastery.com/multi-label-classification-with-deep-learning/
#%%
# Testing the model 
#----------------------------------------------------------------------------------------------------------------------------#

# #train the model on X,y Data
fit = model.fit(X_train , y_train)
print("fit", fit) 

# # #score the model on X,y Data
print("Train score", model.score(X_train , y_train))
print("Test score", model.score(X_test , y_test))

# # #predict the model on X,y Data
predict = model.predict(X_test)
print("predict", predict)

#----------------------------------------------------------------------------------------------------------------------------#
# coef,intercept = model.coef_, model.intercept_; print(f'coef: {coef}, intercept: {intercept}')
# https://realpython.com/linear-regression-in-python/
# y = np.dot(coef,X_train.T)+intercept
#----------------------------------------------------------------------------------------------------------------------------#
#%%
# Pursuing the data analysis
#----------------------------------------------------------------------------------------------------------------------------#
# Metrics : https://scikit-learn.org/stable/modules/model_evaluation.html
# youtube : https://www.youtube.com/watch?v=w_bLGK4Pteo&t=425s

from sklearn.model_selection import (
                                cross_val_score,
                                validation_curve,
                                GridSearchCV,
                                learning_curve
                                )
cv=5
cvs = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy' ) # -> cv-fold type of split for the dataset
# cvs = cross_val_score(KNeighborsRegressor(), X_train, y_train, cv=cv, scoring='accuracy' ) # -> cv-fold type of split for the dataset

val_score = []
n_neighbors = np.arange(1,50)
train_score, val_score = validation_curve(model, X_train, y_train, param_name='n_neighbors', param_range=n_neighbors, cv=cv, scoring='accuracy')
# train_score, val_score = validation_curve(KNeighborsRegressor(), X_train, y_train, param_name='n_neighbors', param_range=n_neighbors, cv=cv, scoring='accuracy')

plt.plot(n_neighbors, val_score.mean(axis=1), label='validation')
plt.plot(n_neighbors, train_score.mean(axis=1), label='train')
plt.xlabel('score')
plt.ylabel('n_neighbors')
plt.legend()

# Overfitting :
# Bon score à l'entrainement et moins bon score à la validation

#Gridsearchcv = https://www.youtube.com/watch?v=w_bLGK4Pteo&t=766s
# sklearn.neighbors.VALID_METRICS['brute']
param_grid = {
            'n_neighbors': np.arange(1,50),
            # 'metric': ['euclidIan'],
            'metric': ['l1','l2','euclidean', 'manhattan','minkowski'],
            }
            
# Defini une grille avec plusieurs estimateurs
grid = GridSearchCV(model, param_grid, cv=cv)
# grid = GridSearchCV(KNeighborsRegressor(), param_grid, cv=cv)

grid.fit(X_train, y_train)
print('grid.best_score_', grid.best_score_)
print('grid.best_params_', grid.best_params_)

print('best model =', grid.best_estimator_)
model = grid.best_estimator_

print('final_score', model.score(X_test, y_test))
#%%
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, model.predict(X_test))

# Courbes d'apprentissage -> possibilité d'amélioration?
# train_size = np.linspace(0, 1.0, 5) -> % -> 20,40,60,80,100%
N, train_score, val_score, = learning_curve(model, X_train, y_train, train_size = np.linspace(0, 1.0, 5))
# N, train_score, val_score, = learning_curve(KNeighborsRegressor(), X_train, y_train, train_size = np.linspace(0, 1.0, 5))
print(f'N: {N}')
plt.plot(N, train_score.mean(axis=1), label='train')
plt.plot(N, val_score.mean(axis=1), label='validation')
plt.xlabel('train_sizes')
plt.legend()
# %%

# comments & advices: https://youtu.be/w_bLGK4Pteo?t=1113

# Docs:
# https://scikit-learn.org/stable/modules/cross_validation.html
# https://scikit-learn.org/stable/modules/grid_search.html#grid-search
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

#----------------------------------------------------------------------------------------------------------------------------#
# best_parameters for KNN

# fit KNeighborsRegressor(metric='l1', n_neighbors=30)
# Train score 0.9344454784123508
# Test score 0.989461381065814
#----------------------------------------------------------------------------------------------------------------------------#


































# %%
#Analogy to validation Curve
# val_score = []
# for n_neighbors in range(1,50):
#     score = cross_val_score(KNeighborsRegressor(k), X_train,y_train, cv).mean()
#     val_score.append(score)

# plt.plot(val_score)
# %%
plt.figure(3,figsize=(8,8))
plt.plot(wavelength_range,coef/np.abs(max(coef)-min(coef)),label = 'result from nnlsqs')
plt.xlabel("wavelength")
plt.ylabel("Normalized_Intensity")
plt.grid();plt.legend()    
plt.title(str(model))
#%%
# Checking  Datas
# 
# '\\\\pollux\\cobiss_YH\\SPARC\\_Usefull_Script_for_further_analysis\\canaux\\FlatSpectrum5.0umP1C3.xlsx'
excelDataArray = pd.read_excel(xlsxpaths[2]).to_numpy()[:,1:]
print(excelDataArray[:,0])
for i in (1003,1013):
    print('a',excelDataArray[i,0])
    print('b',excelDataArray[int(i-1),0])
    print('c',excelDataArray[int(i-2),0])
    print('----------------------')
# for n,i in enumerate(excelDataArray):
#     res = (list(excelDataArray).index(i) == n)
#     print(res)

#%%

# from sklearn.linear_model import TweedieRegressor  ;  model = TweedieRegressor(power=0, alpha=0.5) ##Y=f(X)=f(X1,X2,..,Xn)
# from sklearn.svm          import SVR               ;  model = SVR(C=1000)


# use of Polynomial regression feature <same as> np.polyfit
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=1) #for y = f(X=x1,x2,...,xn)
# poly = PolynomialFeatures(interaction_only=True)
#%%
#Training datas
step=3 # sampling of the X_data datas 
        #for example if step = 5,
        #the Xstep array will be the X_data array with only the index value 0,5,10,15...,last value

Xstep_data, ystep_data = Xr_data[::step,:], yr_data[::step,:]

start, stop, section, stopSection =  10, -23, 28, -25
#take all the sets excepted the middle one which correspond to the 30µW power
# Xstep_train = np.concatenate((Xr_data[start:stop:step,:], Xr_data[int(stop+section):stopSection:step,:]), axis=0)
# ystep_train = np.concatenate((yr_data[start:stop:step,:], yr_data[int(stop+section):stopSection:step,:]), axis=0)
Xstep_train =  Xr_data[start:stop:step,:]
ystep_train =  yr_data[start:stop:step,:]

#Test datas
# Xstep_test ,ystep_test= Xr_data[stop:int(stop+section):step,:], yr_data[stop:int(stop+section):step,:]
Xstep_test ,ystep_test= Xr_data[11:-50:step,:], yr_data[11:-50:step,:]
 

# Xpoly_train , Xpoly_test = poly.fit_transform(Xstep_train), poly.fit_transform(Xstep_test)
X_train , X_test = Xstep_train , Xstep_test
y_train , y_test = ystep_train.T , ystep_test.T 