from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV

from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import  accuracy_score,confusion_matrix,cohen_kappa_score
#from sklearn.preprocessing import StandardScaler as scaler
from sklearn.metrics import r2_score
from sklearn import preprocessing
import pandas as pd
from os.path import isfile
from tsraster.prep import set_common_index, set_df_index,set_df_mindex, image_to_series_simple, seriesToRaster, arrayToRaster, image_to_array
from tsraster import random
import pickle
import numpy as np
from xgboost import XGBRegressor , XGBClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import sklearn.metrics as skmetrics
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence
from skopt.plots import plot_evaluations
from skopt.plots import plot_objective
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import shap
import copy
import pygam
from pygam import LogisticGAM
from contextlib import redirect_stdout
import rpy2
from rpy2 .robjects.packages import importr
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector



def get_data(obj, test_size=0.33,scale=False,stratify=None,groups=None):
    '''
       :param obj: path to csv or name of pandas dataframe  with yX, or list holding dataframes [y,X]
       :param test_size: percentage to hold out for testing (default 0.33)
       :param scale: should data be centered and scaled True or False (default False)
       :param stratify: should the sample be stratified by the dependent value True or False (default None)
       :param groups:  group information defining domain specific stratifications of the samples, ex pixel_id, df.index.get_level_values('index') (default None)
    
       :return: X_train, X_test, y_train, y_test splits
    '''
    
    # read in inputs
    print("input should be csv or pandas dataframe with yX, or [y,X]")
    if str(type(obj)) == "<class 'pandas.core.frame.DataFrame'>":
        df = obj
    
    elif type(obj) == list and len(obj) == 2:
        print('reading in list concat on common index, inner join')
        obj = set_common_index(obj[0], obj[1])
        df = pd.concat([obj[0],obj[1]],axis=1, join='inner') # join Y and X
        df = df.iloc[:,~df.columns.duplicated()]  # remove any repeated columns, take first
    
    elif isfile(obj):
        df = pd.read_csv(obj)
        try:
            set_df_index(df)
        except:
            set_df_mindex(df)
    else:
        print("input format not dataframe, csv, or list")
    
    # remove potential index columns in data 
    df = df.drop(['Unnamed: 0','pixel_id','time'], axis=1,errors ='ignore')  # clear out unknown columns
    
    # check if center and scale
    if scale == True:
        min_max_scaler = preprocessing.MinMaxScaler()
        np_scaled = min_max_scaler.fit_transform(df)
        df = pd.DataFrame(np_scaled)
    
    y = df.iloc[:,0]
    X = df.iloc[:,1:]
    
    # handle stratification by dependent variable
    if stratify == True:
        stratify = y
    else:
        stratify = None
    
    if groups is not None: 
        print('ERROR: need to figure out groups with stratification by y')
        # test train accounting for independent groups
        train_inds, test_inds = next(GroupShuffleSplit().split(X, groups=groups)) 
        X_train, X_test, y_train, y_test = X.iloc[train_inds,:], X.iloc[test_inds,:], y.iloc[train_inds], y.iloc[test_inds]
        
    else:
        # ungrouped test train split 
        X_train, X_test, y_train, y_test = tts(X, y,
                                               test_size=test_size,
                                               stratify=stratify,
                                               random_state=42)
    
    return X_train, X_test, y_train, y_test



def RandomForestReg(X_train, y_train, X_test, y_test, params = {"n_estimators": 100, #determines number of trees to build - more is better, but potentially slows processing
  'criterion' : 'mse', #criterion for measuring the quality of a split (mse or mae)
  'max_depth': 5, #maximum depth of tree - limits tree complexity, may be worth hypertuning 
  'min_samples_split': 2, #sets minimum # of samples per split (serves similar function to min-samples leaf)
  'min_samples_leaf': 10, #sets minimum # of samples per leaf - low values may lead to capturing noise, so may be worth hypertuning
  'min_weight_fraction_leaf': 0, # The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.
  'max_features': 'auto', #max number of features to be considered in a tree (auto means no hard limit)
  'max_leaf_nodes':None, #maximum number of leaf nodes - used to limit model comlexity, similarly to max_depth or min_samples_leaf
  'min_impurity_decrease': 0,
  'min_impurity_split': 1e-7,
  'bootstrap' : True,
  'oob_score' : False, #internal cross-validation - ignore in preference to custom crossval
  'n_jobs' : None, #sets # processors to use, None indicates no limit
  'random_state' : None,
  'verbose' : 0,
  'warm_start': False},
  string_output = False):
    '''
    Conduct random forest regression on training data and test predictive power against test data

    :param X_train: dataframe containing training data features
    :param y_train: dataframe containing training data responses
    :param X_test: dataframe containing test data features
    :param X_train: dataframe containing test data features
    :return: Random Forest Model, dataframe of predicted responses for test dataset, mse of model on test data, r2 of model on test data

    '''

    RF = RandomForestRegressor(n_estimators = params['n_estimators'],
      criterion = params['criterion'],
      max_depth = params['max_depth'],
      min_samples_split = params['min_samples_split'], 
      min_samples_leaf = params['min_samples_leaf'],
      min_weight_fraction_leaf = params['min_weight_fraction_leaf'],
      max_features = params['max_features'],
      max_leaf_nodes = params['max_leaf_nodes'],
      min_impurity_decrease = params['min_impurity_decrease'],
      bootstrap = params['bootstrap'],
      oob_score = params['oob_score'],
      n_jobs = params['n_jobs'],
      random_state = params['random_state'],
      verbose = params['verbose'],
      warm_start = params['warm_start'])

    model = RF.fit(X_train, y_train)
    predict_test = model.predict(X=X_test)

    mse_accuracy = model.score(X_test, y_test)
    r_squared = r2_score(predict_test, y_test)
    
    MSE = model.score(X_test, y_test)
    R_Squared = r2_score(predict_test, y_test)
    
    if string_output == True:
      MSE = ("MSE = {}".format(MSE))
      R_Squared = ("R-Squared = {}".format(R_Squared))

    return RF, predict_test, MSE, R_Squared

def RandomForestClass(X_train, y_train, X_test, y_test):
    '''
    Conduct random forest classification on training data and test predictive power against test data

    :param X_train: dataframe containing training data features
    :param y_train: dataframe containing training data responses
    :param X_test: dataframe containing test data features
    :param X_train: dataframe containing test data features
    :return: Random Forest Model, accuracy of model in predicting test data, mse of model on test data, kappa coefficient, confusion matrix (array)
    '''

    RF = RandomForestClassifier(n_estimators=100,
                               max_depth=10,
                               min_samples_leaf=5,
                               min_samples_split=5,
                               random_state=42,
                               oob_score = True)

    model = RF.fit(X_train, y_train)
    predict_test = model.predict(X=X_test)
    
    test_acc = accuracy_score(y_test, predict_test)
    kappa = cohen_kappa_score(y_test, predict_test)
    confusion = confusion_matrix(y_test, predict_test)
    
    print("Test Accuracy  :: ",test_acc )
    print("Kappa  :: ",kappa )
    
    print("Test Confusion matrix ")
    print(confusion)
    
    return RF,test_acc, kappa, confusion



# Not working correctly
def GradientBoosting(X_train, y_train, X_test, y_test, string_output = False, 
  params = {'loss': 'ls', #determines loss function to be optimized - ls = leas squares regression
  'learning_rate': 0.1, #shrinks the contribution of each tree by the ;learning rate
  'n_estimators': 100, #number of boosting stages to perform
  'subsample': 1.0, #fraction of samples to be used for fitting base learners.  values <1 lead to stochastic gradient boosting, which typically reduces variance but inreases bias
  'criterion': 'friedman_mse', # function used to measure quality of a split.  friedman_mse is an improved version of mse, while mae is also available
  'min_samples_split': 2, #minimum samples required to split a node
  'min_samples_leaf': 1, #minimum samples requred at each leaf node
  'max_depth': 3, #max depth of tree - likely should be hypertuned
  'min_impurity decrease': 0., #a node will be split if that split reduces impurity by at least that amount
  'max_features': None,
  'max_leaf_nodes': None, # determines max number of leaf nodes
   'validation_fraction': 0.1, #proportion of training data to set aside as validation set for early stopping 
   }):
    '''
    Conduct random gradient boosting regression on training data and test predictive power against test data

    :param X_train: dataframe containing training data features
    :param y_train: dataframe containing training data responses
    :param X_test: dataframe containing test data features
    :param X_train: dataframe containing test data features
    :return: gradient boosted regression Model, mse of model on test data, r2 of model on test data
    '''
    GBoost = GradientBoostingRegressor(n_estimators=3000,
                                       learning_rate=0.05,
                                       max_depth=4,
                                       max_features='sqrt',
                                       min_samples_leaf=15,
                                       min_samples_split=10,
                                       loss='huber',
                                       random_state=42)

    model = GBoost.fit(X_train, y_train)
    predict_test = model.predict(X=X_test)

    mse_accuracy = model.score(X_test, y_test)
    r_squared = r2_score(predict_test, y_test)
    
    if string_output == True:
      MSE = ("MSE = {}".format(mse_accuracy))
      R_Squared = ("R-Squared = {}".format(r_squared))
    elif string_output == False:
      MSE = mse_accuracy
      R_Squared = r_squared

    return GBoost, MSE, R_Squared


def ElasticNetModel(X_train, y_train, X_test, y_test, string_output = False, selectedParams = {"alpha":0.5, "l1_ratio":0.7}):
    '''
    Conduct elastic net regression on training data and test predictive power against test data

    :param X_train: dataframe containing training data features
    :param y_train: dataframe containing training data responses
    :return: elastic net model, MSE, R-squared
    '''

    enet = ElasticNet(alpha = selectedParams["alpha"], l1_ratio = selectedParams["l1_ratio"])

    model = enet.fit(X_train, y_train)
    predict_test = model.predict(X=X_test)

    MSE = model.score(X_test, y_test)
    R_Squared = r2_score(predict_test, y_test)
    
    if string_output == True:
      MSE = ("MSE = {}".format(MSE))
      R_Squared = ("R-Squared = {}".format(R_Squared))
    

    return enet, MSE, R_Squared

def ElasticNetCVModel(X_train, y_train, X_test, y_test, string_output = False):
    '''
    Conduct elastic net regression on training data and test predictive power against test data

    :param X_train: dataframe containing training data features
    :param y_train: dataframe containing training data responses
    :return: elastic net model, MSE, R-squared
    '''
    alpha_array = [0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.7, 0.75, 0.8, 0.85, 0.90, 0.95, 0.975, 0.99]
    enet = ElasticNetCV(l1_ratio = alpha_array,
                      n_alphas = 50, 
                      precompute = 'auto',
                      cv = 5,
                      fit_intercept = True)

    model = enet.fit(X_train, y_train)
    predict_test = model.predict(X=X_test)

    mse_accuracy = model.score(X_test, y_test)
    r_squared = r2_score(predict_test, y_test)
    if string_output == True:
      MSE = ("MSE = {}".format(mse_accuracy))
      R_Squared = ("R-Squared = {}".format(r_squared))
    elif string_output == False:
      MSE = mse_accuracy
      R_Squared = r_squared

    return enet, MSE, R_Squared, enet.alphas, enet.l1_ratio


def model_predict(model, new_X):
    '''
    Predicts model based on new data while maintaining the index

    :param model: any fitted model object
    :param new_X: observations to fit model to
    :return: model predictions based on new_X values
    '''
    return  pd.Series(data = model.predict(X=new_X), index = new_X.index)


def model_predict_prob(model, new_X):
    '''
    Predicted class probabilities for model based on new data while maintaining the index

    :param model: any fitted model object
    :param new_X: observations to fit model to
    :return: model class probability predictions based on new_X values
    '''
    return  pd.DataFrame(data = model.predict_proba(X=new_X), index = new_X.index)

def RandomSearch_Tuner(in_model, X_Data, y_Data, in_params, cv=50):
  randomSearch = RandomizedSearchCV(in_model, in_params, cv)
  randomSearch.fit(X_Data, y_Data)
  return randomSearch.best_params_


def elasticNet_2dimTest(combined_Data, target_Data, varsToGroupBy, groupVars, testGroups, DataFields, outPath, params = {"alpha":np.arange(0.05, 1, 0.05), "l1_ratio":np.arange(0.05, 1, 0.005)}, cv = 10):
  '''Conduct elastic net regressions on data, with k-fold cross-validation conducted independently 
      across both years and pixels. 
      Returns mean model MSE and R2 when predicting fire risk at 
      A) locations outside of the training dataset
      B) years outside of the training dataset
      C) locations and years outside of the training dataset

    Returns a list of objects, consisting of:
      0: Combined_Data file with testing/training groups labeled
      1: Target Data file with testing/training groups labeled
      2: summary dataFrame of MSE and R2 for each model run
          (against holdout data representing either novel locations, novel years, or both)
      3: list of elastic net models for use in predicting Fires in further locations/years
      4: list of list of years not used in model training for each run
  '''

  #param combined_Data: explanatory factors to be used in predicting fire risk
  #param target_Data: observed fire occurrences
  #param varsToGroupBy: list of (2) column names from combined_Data & target_Data to be used in creating randomized groups
  #param groupVars: list of (2) desired column names for the resulting randomized groups
  #param testGroups: number of distinct groups into which data sets should be divided (for each of two variables) 
  
  
  #Create randomly assigned groups of equal size by which to separate out subsets of data 
  #by years and by pixels for training and testing to (test against 
  #A) temporally alien, B) spatially alien, and C) completely alien conditions)
  combined_Data, target_Data = random.TestTrain_GroupMaker(combined_Data, target_Data, 
                                                             varsToGroupBy, 
                                                             groupVars, 
                                                             testGroups)



  #get list of group ids, since in cases where group # <10, may not begin at zero
  pixel_testVals = list(set(combined_Data[groupVars[0]].tolist()))
  year_testVals = list(set(combined_Data[groupVars[1]].tolist()))
  
  Models_Summary = pd.DataFrame([], columns = ['Pixels_Years_MSE', 'Pixels_MSE', 'Years_MSE', 
                                             'Pixels_Years_R2', 'Pixels_R2', 'Years_R2'])
  
  #used to create list of model runs
  Models = []
  
  #used to create data for entry as columns into summary DataFrame
  pixels_years_MSEList = []
  pixels_MSEList = []
  years_MSEList = []
  pixels_years_R2List = []
  pixels_R2List = []
  years_R2List = []

  #used to create a list of lists of years that are excluded within each model run
  excluded_Years = []


  #use randomized search to tune hyperparameters on entire dataset
  selectedParams = RandomSearch_Tuner(ElasticNet(), combined_Data.loc[:, DataFields], target_Data['value'], params, cv)

  for x in pixel_testVals:


      for y in year_testVals:
          trainData_X = combined_Data[combined_Data[groupVars[0]] != x]
          trainData_X = trainData_X[trainData_X[groupVars[1]] != y]
          trainData_X = trainData_X.loc[:, DataFields]


          trainData_y = target_Data[target_Data[groupVars[0]] != x]
          trainData_y = trainData_y[trainData_y[groupVars[1]] != y]


          testData_X_pixels_years = combined_Data[combined_Data[groupVars[0]] == x]
          testData_X_pixels_years = testData_X_pixels_years[testData_X_pixels_years[groupVars[1]] == y]
          testData_X_pixels_years = testData_X_pixels_years.loc[:, DataFields]

          testData_X_pixels = combined_Data[combined_Data[groupVars[0]] == x]
          testData_X_pixels = testData_X_pixels[testData_X_pixels[groupVars[1]] != y]
          testData_X_pixels = testData_X_pixels.loc[:, DataFields]

          testData_X_years = combined_Data[combined_Data[groupVars[0]] != x]
          testData_X_years = testData_X_years[testData_X_years[groupVars[1]] == y]
          testData_X_years = testData_X_years.loc[:, DataFields]



          testData_y_pixels_years = target_Data[target_Data[groupVars[0]] == x]
          testData_y_pixels_years = testData_y_pixels_years[testData_y_pixels_years[groupVars[1]] == y]
          

          testData_y_pixels = target_Data[target_Data[groupVars[0]] == x]
          testData_y_pixels = testData_y_pixels[testData_y_pixels[groupVars[1]] != y]
          

          testData_y_years = target_Data[target_Data[groupVars[0]] != x]
          testData_y_years = testData_y_years[testData_y_years[groupVars[1]] == y]
          excluded_Years.append(list(set(testData_y_years[varsToGroupBy[1]].tolist())))
          


          pixels_years_iterOutput = ElasticNetModel(trainData_X, trainData_y['value'], testData_X_pixels_years, testData_y_pixels_years['value'], selectedParams)
          pixels_iterOutput = ElasticNetModel(trainData_X, trainData_y['value'], testData_X_pixels, testData_y_pixels['value'], selectedParams)
          years_iterOutput = ElasticNetModel(trainData_X, trainData_y['value'], testData_X_years, testData_y_years['value'], selectedParams)

          
          Models.append(pixels_years_iterOutput)
          

          pixels_years_MSEList.append(pixels_years_iterOutput[1])
          pixels_MSEList.append(pixels_iterOutput[1])
          years_MSEList.append(years_iterOutput[1])

          pixels_years_R2List.append(pixels_years_iterOutput[2])
          pixels_R2List.append(pixels_iterOutput[2])
          years_R2List.append(years_iterOutput[2])

  
  
  #combine MSE and R2 Lists into single DataFrame
  Models_Summary['Pixels_Years_MSE'] = pixels_years_MSEList
  Models_Summary['Pixels_MSE'] = pixels_MSEList
  Models_Summary['Years_MSE'] = years_MSEList
  
  Models_Summary['Pixels_Years_R2'] = pixels_years_R2List
  Models_Summary['Pixels_R2'] = pixels_R2List
  Models_Summary['Years_R2'] = years_R2List
  
  
  print("pixels_Years MSE Overall: ", sum(pixels_years_MSEList)/len(pixels_years_MSEList))
  print("pixels_Years R2 Overall: ", sum(pixels_years_R2List)/len(pixels_years_R2List))
  #print("pixels_Years R2 iterations: ", pixels_years_R2List)
  print("\n")
  print("pixels MSE Overall: ", sum(pixels_MSEList)/len(pixels_MSEList))
  print("pixels R2 Overall: ", sum(pixels_R2List)/len(pixels_R2List))
  #print("pixels R2 iterations: ", pixels_R2List)
  print("\n")
  print("years MSE Overall: ", sum(years_MSEList)/len(years_MSEList))
  print("years R2 Overall: ", sum(years_R2List)/len(years_R2List))
  #print("years R2 iterations: ", years_R2List)
  print("\n")
  
  pickling_on = open(outPath + "elasticNet_2dim.pickle", "wb")
  pickle.dump([combined_Data, target_Data, Models_Summary, Models, excluded_Years, selectedParams], pickling_on)
  pickling_on.close

  Models_Summary.to_csv(outPath + "Model_Summary.csv")

  return combined_Data, target_Data, Models_Summary, Models, excluded_Years, selectedParams


def XGBoostReg_2dimTest(combined_Data, target_Data, varsToGroupBy, groupVars, testGroups, 
                        DataFields, outPath, 
                        params = None, cv = 10):

    combined_Data, target_Data = random.TestTrain_GroupMaker(combined_Data, target_Data, 
                                                             varsToGroupBy, 
                                                             groupVars, 
                                                             testGroups)

    #get list of group ids, since in cases where group # <10, may not begin at zero
    pixel_testVals = list(set(combined_Data[groupVars[0]].tolist()))
    year_testVals = list(set(combined_Data[groupVars[1]].tolist()))

    Models_Summary = pd.DataFrame([], columns = ['Pixels_Years_MSE', 'Pixels_MSE', 'Years_MSE', 
                                             'Pixels_Years_R2', 'Pixels_R2', 'Years_R2'])

    #used to create list of model runs
    Models = []
  
  #used to create data for entry as columns into summary DataFrame
    pixels_years_MSEList = []
    pixels_MSEList = []
    years_MSEList = []
    pixels_years_R2List = []
    pixels_R2List = []
    years_R2List = []

  #used to create a list of lists of years that are excluded within each model run
    excluded_Years = []


     
    selectedParams = params
    
    for x in pixel_testVals:


        for y in year_testVals:
            trainData_X = combined_Data[combined_Data[groupVars[0]] != x]
            trainData_X = trainData_X[trainData_X[groupVars[1]] != y]
            trainData_X = trainData_X.loc[:, DataFields]


            trainData_y = target_Data[target_Data[groupVars[0]] != x]
            trainData_y = trainData_y[trainData_y[groupVars[1]] != y]


            testData_X_pixels_years = combined_Data[combined_Data[groupVars[0]] == x]
            testData_X_pixels_years = testData_X_pixels_years[testData_X_pixels_years[groupVars[1]] == y]
            testData_X_pixels_years = testData_X_pixels_years.loc[:, DataFields]

            testData_X_pixels = combined_Data[combined_Data[groupVars[0]] == x]
            testData_X_pixels = testData_X_pixels[testData_X_pixels[groupVars[1]] != y]
            testData_X_pixels = testData_X_pixels.loc[:, DataFields]

            testData_X_years = combined_Data[combined_Data[groupVars[0]] != x]
            testData_X_years = testData_X_years[testData_X_years[groupVars[1]] == y]
            testData_X_years = testData_X_years.loc[:, DataFields]



            testData_y_pixels_years = target_Data[target_Data[groupVars[0]] == x]
            testData_y_pixels_years = testData_y_pixels_years[testData_y_pixels_years[groupVars[1]] == y]


            testData_y_pixels = target_Data[target_Data[groupVars[0]] == x]
            testData_y_pixels = testData_y_pixels[testData_y_pixels[groupVars[1]] != y]


            testData_y_years = target_Data[target_Data[groupVars[0]] != x]
            testData_y_years = testData_y_years[testData_y_years[groupVars[1]] == y]
            excluded_Years.append(list(set(testData_y_years[varsToGroupBy[1]].tolist())))

            pixels_years_iterOutput = XGBoostModel(trainData_X, trainData_y['value'], testData_X_pixels_years, testData_y_pixels_years['value'], selectedParams)
            pixels_iterOutput = XGBoostModel(trainData_X, trainData_y['value'], testData_X_pixels, testData_y_pixels['value'], selectedParams)
            years_iterOutput = XGBoostModel(trainData_X, trainData_y['value'], testData_X_years, testData_y_years['value'], selectedParams)


            Models.append(pixels_years_iterOutput)


            pixels_years_MSEList.append(pixels_years_iterOutput[1])
            pixels_MSEList.append(pixels_iterOutput[1])
            years_MSEList.append(years_iterOutput[1])

            pixels_years_R2List.append(pixels_years_iterOutput[2])
            pixels_R2List.append(pixels_iterOutput[2])
            years_R2List.append(years_iterOutput[2])
        
    #combine MSE and R2 Lists into single DataFrame
    Models_Summary['Pixels_Years_MSE'] = pixels_years_MSEList
    Models_Summary['Pixels_MSE'] = pixels_MSEList
    Models_Summary['Years_MSE'] = years_MSEList

    Models_Summary['Pixels_Years_R2'] = pixels_years_R2List
    Models_Summary['Pixels_R2'] = pixels_R2List
    Models_Summary['Years_R2'] = years_R2List


    print("pixels_Years MSE Overall: ", sum(pixels_years_MSEList)/len(pixels_years_MSEList))
    print("pixels_Years R2 Overall: ", sum(pixels_years_R2List)/len(pixels_years_R2List))
    #print("pixels_Years R2 iterations: ", pixels_years_R2List)
    print("\n")
    print("pixels MSE Overall: ", sum(pixels_MSEList)/len(pixels_MSEList))
    print("pixels R2 Overall: ", sum(pixels_R2List)/len(pixels_R2List))
    #print("pixels R2 iterations: ", pixels_R2List)
    print("\n")
    print("years MSE Overall: ", sum(years_MSEList)/len(years_MSEList))
    print("years R2 Overall: ", sum(years_R2List)/len(years_R2List))
    #print("years R2 iterations: ", years_R2List)
    print("\n")

    pickling_on = open(outPath + "XGBoost_2dim.pickle", "wb")
    pickle.dump([combined_Data, target_Data, Models_Summary, Models, excluded_Years, selectedParams], pickling_on)
    pickling_on.close

    Models_Summary.to_csv(outPath + "Model_Summary_XGBOOST.csv")

    return combined_Data, target_Data, Models_Summary, Models, excluded_Years, selectedParams
 

def zeroMasker(row):
    ''' used as apply statement for masking in elastic_YearPredictor
    :return: 0 if cell is masked out, PredRisk otherwise
    '''
    if row['mask'] == 0:
        return 0
    else:
        return row['PredRisk']

def elastic_YearPredictor(combined_Data_Training, target_Data_Training, preMasked_Data_Path, outPath, year_List, periodLen, DataFields, mask, params):
    '''annually predict fire risk- train model on combined_Data across all available years except year of interest
    save resulting predictions as csv and as tif to location 'outPath'
    
    :param combined_Data_Training: dataFrame including all desired explanatory factors 
            across all locations & years to be used in training model
    :param target_Data_Training: dataFrame including observed fire occurrences 
            across all locations & years to be used in training model
    :param preMasked_Data_Path: file path to location of files to use in predicting fire risk 
                    (note - these files should not have undergone Poisson disk masking)
    :param outPath: desired output location for predicted fire risk files (csv, pickle, and tif)
    :param year_List: list of years for which predictions are desired
    :param Datafields: list of explanatory factors to be intered into model
    :param mask: filepath of raster mask to be used in masking output predictions, 
            and as an example raster for choosing array shape and projections for .tif output files
    :param params: parameters for elastic net regression (presumably developed from 2dimCrossval)
    :return:  returns a list of all models, accompanied by a list of years being predicted 
            - note - return output is equivalent to data exported as models.pickle
    '''
    
    model_List = []
    
    for iterYear in year_List:
        combined_Data_iter_train = combined_Data_Training[combined_Data_Training['year'] != iterYear]
        combined_Data_iter_train = combined_Data_iter_train.loc[:, DataFields]
        
        target_Data_iter_train = target_Data_Training[target_Data_Training['year'] != iterYear]
        
        
        elastic_iter_Model = ElasticNet(l1_ratio = params['l1_ratio'], alpha = params['alpha'])
        
        elastic_iter_Fit = elastic_iter_Model.fit(combined_Data_iter_train, target_Data_iter_train['value'])
        
        
        
        #seriesToRaster(predict_iter, templateRasterPath, outPath + "Pred_FireRisk_" + str(iterYear) + ".tif")

        full_X = pd.read_csv(preMasked_Data_Path + "CD_" + str(iterYear) + "_" + str(iterYear + periodLen - 1) + ".csv")
        full_X = full_X.loc[:, DataFields]
        
        data = elastic_iter_Fit.predict(X=full_X)
        print(data)
        data = pd.DataFrame(data, columns = ['PredRisk'])
        index_mask = image_to_series_simple(mask)
        data['mask'] = index_mask
        data['PredRisk_Masked'] = data.apply(zeroMasker, axis =1)
        
        data.to_csv(outPath + "Pred_" + str(iterYear) + ".csv")
        
        #output predicted risk as tiff
        seriesToRaster(data['PredRisk_Masked'], mask, outPath + "Pred_" + str(iterYear) + "_" + str(iterYear + periodLen - 1) + "_elastic.tif", noData = -9999)
        
        model_List.append([elastic_iter_Fit])
        
    pickling_on = open(outPath + "elastic_models.pickle", "wb")
    pickle.dump([model_List, year_List], pickling_on)
    pickling_on.close
        
        
    return model_List, year_List

def randomForestReg_YearPredictor(combined_Data_Training, target_Data_Training, preMasked_Data_Path, outPath, year_List, periodLen, DataFields, mask, params):
    '''annually predict fire risk- train model on combined_Data across all available years except year of interest
    save resulting predictions as csv and as tif to location 'outPath'
    
    :param combined_Data_Training: dataFrame including all desired explanatory factors 
            across all locations & years to be used in training model
    :param target_Data_Training: dataFrame including observed fire occurrences 
            across all locations & years to be used in training model
    :param preMasked_Data_Path: file path to location of files to use in predicting fire risk 
                    (note - these files should not have undergone Poisson disk masking)
    :param outPath: desired output location for predicted fire risk files (csv, pickle, and tif)
    :param year_List: list of years for which predictions are desired
    :param Datafields: list of explanatory factors to be intered into model
    :param mask: filepath of raster mask to be used in masking output predictions, 
            and as an example raster for choosing array shape and projections for .tif output files
    :param params: parameters for random forest regression (presumably developed from 2dimCrossval)
    :return:  returns a list of all models, accompanied by a list of years being predicted 
            - note - return output is equivalent to data exported as models.pickle
    '''
    
    model_List = []
    
    for iterYear in year_List:
        combined_Data_iter_train = combined_Data_Training[combined_Data_Training['year'] != iterYear]
        combined_Data_iter_train = combined_Data_iter_train.loc[:, DataFields]
        
        target_Data_iter_train = target_Data_Training[target_Data_Training['year'] != iterYear]
        
        
        iter_Model = RandomForestRegressor(n_estimators = params["n_estimators"], criterion = params["criterion"],
          max_depth = params["max_depth"], min_samples_split = params['min_samples_split'], min_samples_leaf = params['min_samples_leaf'],
        min_weight_fraction_leaf = params['min_weight_fraction_leaf'], max_features = params['max_features'], max_leaf_nodes = params['max_leaf_nodes'],
        min_impurity_decrease = params['min_impurity_decrease'], bootstrap = params['bootstrap'], oob_score = params['oob_score'], n_jobs = params['n_jobs'],
        random_state = params['random_state'], verbose = params['verbose'], warm_start = params['warm_start'])
        
        iter_Fit = iter_Model.fit(combined_Data_iter_train, target_Data_iter_train['value'])
        
        
        
        #seriesToRaster(predict_iter, templateRasterPath, outPath + "Pred_FireRisk_" + str(iterYear) + ".tif")

        full_X = pd.read_csv(preMasked_Data_Path + "CD_" + str(iterYear) + "_" + str(iterYear + periodLen - 1) + ".csv")
        full_X = full_X.loc[:, DataFields]
        
        data = iter_Fit.predict(X=full_X)
        print(data)
        data = pd.DataFrame(data, columns = ['PredRisk'])
        index_mask = image_to_series_simple(mask)
        data['mask'] = index_mask
        data['PredRisk_Masked'] = data.apply(zeroMasker, axis =1)
        
        data.to_csv(outPath + "Pred_" + str(iterYear) + ".csv")
        
        #output predicted risk as tiff
        seriesToRaster(data['PredRisk_Masked'], mask, outPath + "Pred_" +str(iterYear) + "_" + str(iterYear + periodLen - 1) + "RFreg.tif")
        
        model_List.append([iter_Fit])
        
    pickling_on = open(outPath + "models.pickle", "wb")
    pickle.dump([model_List, year_List], pickling_on)
    pickling_on.close
        
        
    return model_List, year_List

def XGBoostModel(X_train, y_train, X_test, y_test, string_output = False, 
                 selectedParams = {"learning_rate":0.1, "max_features": 5, 
                                   "min_samples_split":15, "min_samples_leaf":33}):
    '''
    Conduct elastic net regression on training data and test predictive power against test data

    :param X_train: dataframe containing training data features
    :param y_train: dataframe containing training data responses
    :return: elastic net model, MSE, R-squared
    '''

    xgbr = XGBRegressor(learning_rate = selectedParams["learning_rate"], 
                        max_features = selectedParams['max_features'],
                       min_samples_split = selectedParams['min_samples_split'],
                       min_samples_leaf = selectedParams['min_samples_leaf'])

    model = xgbr.fit(X_train, y_train.values)  #must convert y_train to values to prevent an erroneous error warning
    predict_test = model.predict(data = X_test)

    MSE = model.score(X_test, y_test)
    R_Squared = r2_score(predict_test, y_test)
    
    if string_output == True:
      MSE = ("MSE = {}".format(MSE))
      R_Squared = ("R-Squared = {}".format(R_Squared))
    

    return xgbr, MSE, R_Squared, predict_test


def XGBoostReg_2dimTest(combined_Data, target_Data, varsToGroupBy, groupVars, testGroups, 
                        DataFields, outPath, 
                        params = None, cv = 10):

    combined_Data, target_Data = random.TestTrain_GroupMaker(combined_Data, target_Data, 
                                                             varsToGroupBy, 
                                                             groupVars, 
                                                             testGroups)

    #get list of group ids, since in cases where group # <10, may not begin at zero
    pixel_testVals = list(set(combined_Data[groupVars[0]].tolist()))
    year_testVals = list(set(combined_Data[groupVars[1]].tolist()))

    Models_Summary = pd.DataFrame([], columns = ['Pixels_Years_MSE', 'Pixels_MSE', 'Years_MSE', 
                                             'Pixels_Years_R2', 'Pixels_R2', 'Years_R2'])

    #used to create list of model runs
    Models = []
  
  #used to create data for entry as columns into summary DataFrame
    pixels_years_MSEList = []
    pixels_MSEList = []
    years_MSEList = []
    pixels_years_R2List = []
    pixels_R2List = []
    years_R2List = []

  #used to create a list of lists of years that are excluded within each model run
    excluded_Years = []


     
    selectedParams = params
    
    for x in pixel_testVals:


        for y in year_testVals:
            trainData_X = combined_Data[combined_Data[groupVars[0]] != x]
            trainData_X = trainData_X[trainData_X[groupVars[1]] != y]
            trainData_X = trainData_X.loc[:, DataFields]


            trainData_y = target_Data[target_Data[groupVars[0]] != x]
            trainData_y = trainData_y[trainData_y[groupVars[1]] != y]


            testData_X_pixels_years = combined_Data[combined_Data[groupVars[0]] == x]
            testData_X_pixels_years = testData_X_pixels_years[testData_X_pixels_years[groupVars[1]] == y]
            testData_X_pixels_years = testData_X_pixels_years.loc[:, DataFields]

            testData_X_pixels = combined_Data[combined_Data[groupVars[0]] == x]
            testData_X_pixels = testData_X_pixels[testData_X_pixels[groupVars[1]] != y]
            testData_X_pixels = testData_X_pixels.loc[:, DataFields]

            testData_X_years = combined_Data[combined_Data[groupVars[0]] != x]
            testData_X_years = testData_X_years[testData_X_years[groupVars[1]] == y]
            testData_X_years = testData_X_years.loc[:, DataFields]



            testData_y_pixels_years = target_Data[target_Data[groupVars[0]] == x]
            testData_y_pixels_years = testData_y_pixels_years[testData_y_pixels_years[groupVars[1]] == y]


            testData_y_pixels = target_Data[target_Data[groupVars[0]] == x]
            testData_y_pixels = testData_y_pixels[testData_y_pixels[groupVars[1]] != y]


            testData_y_years = target_Data[target_Data[groupVars[0]] != x]
            testData_y_years = testData_y_years[testData_y_years[groupVars[1]] == y]
            excluded_Years.append(list(set(testData_y_years[varsToGroupBy[1]].tolist())))

            pixels_years_iterOutput = XGBoostModel(trainData_X, trainData_y['value'], testData_X_pixels_years, testData_y_pixels_years['value'], selectedParams)
            pixels_iterOutput = XGBoostModel(trainData_X, trainData_y['value'], testData_X_pixels, testData_y_pixels['value'], selectedParams)
            years_iterOutput = XGBoostModel(trainData_X, trainData_y['value'], testData_X_years, testData_y_years['value'], selectedParams)


            Models.append(pixels_years_iterOutput)


            pixels_years_MSEList.append(pixels_years_iterOutput[1])
            pixels_MSEList.append(pixels_iterOutput[1])
            years_MSEList.append(years_iterOutput[1])

            pixels_years_R2List.append(pixels_years_iterOutput[2])
            pixels_R2List.append(pixels_iterOutput[2])
            years_R2List.append(years_iterOutput[2])
        
    #combine MSE and R2 Lists into single DataFrame
    Models_Summary['Pixels_Years_MSE'] = pixels_years_MSEList
    Models_Summary['Pixels_MSE'] = pixels_MSEList
    Models_Summary['Years_MSE'] = years_MSEList

    Models_Summary['Pixels_Years_R2'] = pixels_years_R2List
    Models_Summary['Pixels_R2'] = pixels_R2List
    Models_Summary['Years_R2'] = years_R2List


    print("pixels_Years MSE Overall: ", sum(pixels_years_MSEList)/len(pixels_years_MSEList))
    print("pixels_Years R2 Overall: ", sum(pixels_years_R2List)/len(pixels_years_R2List))
    #print("pixels_Years R2 iterations: ", pixels_years_R2List)
    print("\n")
    print("pixels MSE Overall: ", sum(pixels_MSEList)/len(pixels_MSEList))
    print("pixels R2 Overall: ", sum(pixels_R2List)/len(pixels_R2List))
    #print("pixels R2 iterations: ", pixels_R2List)
    print("\n")
    print("years MSE Overall: ", sum(years_MSEList)/len(years_MSEList))
    print("years R2 Overall: ", sum(years_R2List)/len(years_R2List))
    #print("years R2 iterations: ", years_R2List)
    print("\n")

    pickling_on = open(outPath + "XGBoost_2dim.pickle", "wb")
    pickle.dump([combined_Data, target_Data, Models_Summary, Models, excluded_Years, selectedParams], pickling_on)
    pickling_on.close

    Models_Summary.to_csv(outPath + "Model_Summary_XGBOOST.csv")

    return combined_Data, target_Data, Models_Summary, Models, excluded_Years, selectedParams
 


def XGBoostReg_YearPredictor(combined_Data_Training, target_Data_Training, preMasked_Data_Path, outPath, year_List, periodLen, DataFields, mask, params = {'eta': 0.3, #step size shrinkage used in updates to prevent overfitting (also called learning_rate)
         'n_estimators': 100}):
    '''annually predict fire risk- train model on combined_Data across all available years except year of interest
    save resulting predictions as csv and as tif to location 'outPath'
    
    :param combined_Data_Training: dataFrame including all desired explanatory factors 
            across all locations & years to be used in training model
    :param target_Data_Training: dataFrame including observed fire occurrences 
            across all locations & years to be used in training model
    :param preMasked_Data_Path: file path to location of files to use in predicting fire risk 
                    (note - these files should not have undergone Poisson disk masking)
    :param outPath: desired output location for predicted fire risk files (csv, pickle, and tif)
    :param year_List: list of years for which predictions are desired
    :param Datafields: list of explanatory factors to be intered into model
    :param mask: filepath of raster mask to be used in masking output predictions, 
            and as an example raster for choosing array shape and projections for .tif output files
    :param params: parameters for random forest regression (presumably developed from 2dimCrossval)
    :return:  returns a list of all models, accompanied by a list of years being predicted 
            - note - return output is equivalent to data exported as models.pickle
    '''
    
    model_List = []
    
    for iterYear in year_List:
        combined_Data_iter_train = combined_Data_Training[combined_Data_Training['year'] != iterYear]
        combined_Data_iter_train = combined_Data_iter_train.loc[:, DataFields]
        
        target_Data_iter_train = target_Data_Training[target_Data_Training['year'] != iterYear]
        
        
        iter_Model = XGBRegressor(learning_rate = params["eta"], n_estimators = params["n_estimators"])
        
        iter_Fit = iter_Model.fit(combined_Data_iter_train, target_Data_iter_train['value'].values)
        
        
        
        #seriesToRaster(predict_iter, templateRasterPath, outPath + "Pred_FireRisk_" + str(iterYear) + ".tif")

        full_X = pd.read_csv(preMasked_Data_Path + "CD_" + str(iterYear) + "_" + str(iterYear + periodLen - 1) + ".csv")
        full_X = full_X.loc[:, DataFields]

        data = iter_Fit.predict(full_X)
        print(data)
        data = pd.DataFrame(data, columns = ['PredRisk'])
        index_mask = image_to_series_simple(mask)
        data['mask'] = index_mask
        data['PredRisk_Masked'] = data.apply(zeroMasker, axis =1)
        
        data.to_csv(outPath + "Pred_" + str(iterYear) + ".csv")
        
        #output predicted risk as tiff
        seriesToRaster(data['PredRisk_Masked'], mask, outPath + "Pred_" + str(iterYear) + "_" + str(iterYear + periodLen - 1) + "XGBoost.tif")
        
        model_List.append([iter_Fit])
        
    pickling_on = open(outPath + "models.pickle", "wb")
    pickle.dump([model_List, year_List], pickling_on)
    pickling_on.close
        
        
    return model_List, year_List


def GBoost_skopt(X, y, outPath, n_calls = 50, n_estimators = 100, random_state = 0, n_jobs = 5):
    '''conducts hyperparameter tuning using scikit-optimize
    param X: multidimensional array of explanatory factors
    param y: vector of response variable
    n_calls: number of calls (evaluations) for gp_minimize
    n_estimators: number of estimators (boosting stages) for gradientboosting
    random state: random seed
    n_jobs = number of jobs for parallellization
    '''
    
    space  = [Integer(1, 5, name='max_depth'),
      Real(10**-5, 10**0, "log-uniform", name='learning_rate'),
      Integer(1, X.shape[1], name='max_features'),
      Integer(2, 100, name='min_samples_split'),
      Integer(1, 100, name='min_samples_leaf')]

    @use_named_args(space)
    def objective(**hyperParams):
        reg.set_params(**hyperParams)
        return -np.mean(cross_val_score(reg, X, y, cv = 5, n_jobs = n_jobs, scoring = "neg_mean_absolute_error"))
            #may want to replace with different scorer - may even include custom scorers using make_scorer (https://scikit-learn.org/stable/modules/model_evaluation.html#scoring)


    print(n_calls)
    reg = GradientBoostingRegressor(n_estimators = n_estimators, random_state = random_state)
    res_gp = gp_minimize(objective, space, n_calls= n_calls, random_state= random_state)
    
    outString = """"Best Score_Hyperopt = %.4f
    
    Best hyperparameters:
    - max_depth=%d
    - learning_rate=%.6f
    - max_features=%d
    - min_samples_split=%d
    - min_samples_leaf=%d""" % (res_gp.fun, res_gp.x[0], res_gp.x[1], 
                                res_gp.x[2], res_gp.x[3], 
                                res_gp.x[4])
    print(outString)
    
    text_file = open(outPath + "HyperParams.txt", "w+")
    text_file.write(outString)
    text_file.close()
    
    fig, ax = plt.subplots(figsize=(4,4))
    plot_convergence(res_gp, ax=ax)
    plt.tight_layout()
    plt.savefig(outPath + 'forest_convergence_test.png')
    
    
   
    plt.clf()
    fig = plot_evaluations(res_gp, dimensions = [space[0].name, space[1].name, space[2].name, space[3].name, space[4].name])
    plt.savefig(outPath + 'forest_evaluations.png')
    
    plt.clf()
    fig = plot_objective(res_gp, dimensions = [space[0].name, space[1].name, space[2].name, space[3].name, space[4].name])
    plt.savefig(outPath + 'objectives.png')
    
    out_hyperParams = {'n_estimators': n_estimators, 'max_depth':res_gp.x[0], 'learning_rate':res_gp.x[1], 
                       'max_features': res_gp.x[2], 'min_samples_split': res_gp.x[3],'min_samples_leaf': res_gp.x[4]}
    
    return (out_hyperParams)

def GBoost_skopt_classifier(X, y, outPath, n_calls = 50, n_estimators = 100, random_state = 0, n_jobs = 5):
    
    space  = [Integer(1, 5, name='max_depth'),
          Real(10**-5, 10**0, "log-uniform", name='learning_rate'),
          Integer(1, X.shape[1], name='max_features'),
          Integer(2, 100, name='min_samples_split'),
          Integer(1, 100, name='min_samples_leaf')]

    @use_named_args(space)
    def objective(**hyperParams):
        reg.set_params(**hyperParams)
        return -np.mean(cross_val_score(reg, X, y, cv = 5, n_jobs = n_jobs, scoring = "balanced_accuracy"))


    print(n_calls)
    reg = GradientBoostingClassifier(n_estimators = n_estimators, random_state = random_state)
    res_gp = gp_minimize(objective, space, n_calls= n_calls, random_state= random_state)
    
    outString = """"Best Score_Hyperopt = %.4f
    
    Best hyperparameters:
    - max_depth=%d
    - learning_rate=%.6f
    - max_features=%d
    - min_samples_split=%d
    - min_samples_leaf=%d""" % (res_gp.fun, res_gp.x[0], res_gp.x[1], 
                                res_gp.x[2], res_gp.x[3], 
                                res_gp.x[4])
    print(outString)
    
    text_file = open(outPath + "HyperParams.txt", "w+")
    text_file.write(outString)
    text_file.close()
    
    fig, ax = plt.subplots(figsize=(4,4))
    plot_convergence(res_gp, ax=ax)
    plt.tight_layout()
    plt.savefig(outPath + 'forest_convergence_test.png')
    
    
   
    plt.clf()
    fig = plot_evaluations(res_gp, dimensions = [space[0].name, space[1].name, space[2].name, space[3].name, space[4].name])
    plt.savefig(outPath + 'forest_evaluations.png')
    
    plt.clf()
    fig = plot_objective(res_gp, dimensions = [space[0].name, space[1].name, space[2].name, space[3].name, space[4].name])
    plt.savefig(outPath + 'objectives.png')
    
    out_hyperParams = {'n_estimators': n_estimators, 'max_depth':res_gp.x[0], 'learning_rate':res_gp.x[1], 
                       'max_features': res_gp.x[2], 'min_samples_split': res_gp.x[3],'min_samples_leaf': res_gp.x[4]}
    
    return (out_hyperParams)



def XGBoostModel_Class(X_train, y_train, X_test, y_test, string_output = False, 
                 params = {"learning_rate":0.1, "max_features": 5, 
                                   "min_samples_split":15, "min_samples_leaf":33}):
    '''
    Conduct elastic net regression on training data and test predictive power against test data

    :param X_train: dataframe containing training data features
    :param y_train: dataframe containing training data responses
    :return: elastic net model, MSE, R-squared
    '''

    xgbr = XGBClassifier(learning_rate = params["learning_rate"], 
                        max_features = params['max_features'],
                       min_samples_split = params['min_samples_split'],
                       min_samples_leaf = params['min_samples_leaf'])

    model = xgbr.fit(X_train, y_train.values)  #must convert y_train to values to prevent an erroneous error warning
    predict_test = model.predict(data = X_test)
    predict_risk = model.predict_proba(X_test)
    predict_risk =  predict_risk[:, 1]

    MSE = model.score(X_test, y_test)
    R_Squared = r2_score(predict_test, y_test)
    
    Accuracy = skmetrics.accuracy_score(y_test, predict_test)
    BalancedAccuracy = skmetrics.balanced_accuracy_score(y_test, predict_test)
    f1_binary = skmetrics.f1_score(y_test, predict_test, average = 'binary')
    f1_macro = skmetrics.f1_score(y_test, predict_test, average = 'macro')
    f1_micro = skmetrics.f1_score(y_test, predict_test, average = 'micro')
    log_loss = skmetrics.log_loss(y_test, predict_test, labels = [0,1])
    recall_binary = skmetrics.recall_score(y_test, predict_test, average = 'binary')
    recall_macro = skmetrics.recall_score(y_test, predict_test, average = 'macro')
    recall_micro = skmetrics.recall_score(y_test, predict_test, average = 'micro')
    jaccard_binary = skmetrics.jaccard_score(y_test, predict_test,average = 'binary')
    jaccard_macro = skmetrics.jaccard_score(y_test, predict_test, average = 'macro')
    jaccard_micro = skmetrics.jaccard_score(y_test, predict_test, average = 'micro')
    roc_auc_macro = skmetrics.roc_auc_score(y_test, predict_risk, average = 'macro')
    roc_auc_micro = skmetrics.roc_auc_score(y_test, predict_risk, average = 'micro')
    average_precision = skmetrics.roc_auc_score(y_test, predict_risk)
    
    explainerXGB = shap.TreeExplainer(model)
    #shap_values_XGB_test = explainerXGB.shap_values(X_test)
    shap_values_train = explainerXGB.shap_values(X_train)
    
    #df_shap_XGB_test = pd.DataFrame(shap_values_XGB_test, columns = x_test.columns.values)
    #df_shap_train = pd.DataFrame(shap_values_train, columns = x_train.columns.values)
    
    if string_output == True:
      MSE = ("MSE = {}".format(MSE))
      R_Squared = ("R-Squared = {}".format(R_Squared))
    
    #feature importances:
    gain = pd.DataFrame([ model.get_booster().get_score(importance_type = 'gain')]) #average gain
    t_gain = pd.DataFrame([ model.get_booster().get_score(importance_type = 'total_gain')])  #total gain
    cover = pd.DataFrame([ model.get_booster().get_score(importance_type = 'cover')]) # coverage - mean quantity of observations concerned by a feature
    t_cover = pd.DataFrame([model.get_booster().get_score(importance_type = 'total_cover')]) # total quantity of observations concerned by a feature
    weight = pd.DataFrame([ model.get_booster().get_score(importance_type = 'weight')]) # % representing the number of times a particular feature occurs in the trees of the model
    f_importances = {'gain': gain, 't_gain': t_gain, 'cover':cover, 't_cover': t_cover, 'weight':weight}
        

    return xgbr, MSE, R_Squared, f1_binary, f1_macro, f1_micro, log_loss, recall_binary, recall_macro, recall_micro, jaccard_binary, jaccard_macro, jaccard_micro, roc_auc_macro, roc_auc_micro, average_precision, f_importances, predict_test, shap_values_train 

def XGBoostClass_2dimTest(combined_Data, target_Data, varsToGroupBy, groupVars, testGroups, 
                        DataFields, outPath, 
                        params = None, cv = 10):

    combined_Data, target_Data = random.TestTrain_GroupMaker(combined_Data, target_Data, 
                                                             varsToGroupBy, 
                                                             groupVars, 
                                                             testGroups)

    #get list of group ids, since in cases where group # <10, may not begin at zero
    pixel_testVals = list(set(combined_Data[groupVars[0]].tolist()))
    year_testVals = list(set(combined_Data[groupVars[1]].tolist()))

    Models_Summary = pd.DataFrame([], columns = ['Pixels_Years_MSE', 'Pixels_MSE', 'Years_MSE', 
                                             'Pixels_Years_R2', 'Pixels_R2', 'Years_R2',
                                                'Pixels_Years_Accuracy', 'Pixels_Accuracy', 'Years_Accuracy',
                                                'Pixels_Years_BalancedAccuracy', 'Pixels_BalancedAccuracy', 'Years_BalancedAccuracy',
                                                'Pixels_Years_F1_binary', 'Pixels_F1_binary', 'Years_F1_binary',
                                                'Pixels_Years_F1_Macro', 'Pixels_F1_Macro', 'Years_F1_Macro',
                                                'Pixels_Years_F1_Micro', 'Pixels_F1_Micro', 'Years_F1_Micro',
                                                'Pixels_Years_logLoss', 'Pixels_logLoss', 'Years_logLoss',
                                                'Pixels_Years_recall_binary', 'Pixels_recall_binary', 'Years_recall_binary',
                                                'Pixels_Years_recall_Macro', 'Pixels_recall_Macro', 'Years_recall_Macro',
                                                'Pixels_Years_recall_Micro', 'Pixels_recall_Micro', 'Years_recall_Micro',
                                                'Pixels_Years_jaccard_binary', 'Pixels_jaccard_binary', 'Years_jaccard_binary',
                                                'Pixels_Years_jaccard_Macro', 'Pixels_jaccard_Macro', 'Years_jaccard_Macro',
                                                'Pixels_Years_jaccard_Micro', 'Pixels_jaccard_Micro', 'Years_jaccard_Micro',
                                                'Pixels_Years_roc_auc_Macro', 'Pixels_jaccard_roc_auc_Macro', 'Years_jaccard_roc_auc_Macro',
                                                 'Pixels_Years_roc_auc_Micro', 'Pixels_jaccard_roc_auc_Micro', 'Years_jaccard_roc_auc_Micro',
                                                'average_precision'])

    #used to create list of model runs
    Models = []
    
    
  
  #used to create data for entry as columns into summary DataFrame
    pixels_years_MSEList = []
    pixels_MSEList = []
    years_MSEList = []
    pixels_years_R2List = []
    pixels_R2List = []
    years_R2List = []
    
    pixels_years_F1_binaryList = []
    pixels_F1_binaryList = []
    years_F1_binaryList = []
    
    pixels_years_F1_MacroList = []
    pixels_F1_MacroList = []
    years_F1_MacroList = []
    
    pixels_years_F1_MicroList = []
    pixels_F1_MicroList = []
    years_F1_MicroList = []
    
    pixels_years_logLossList = []
    pixels_logLossList = []
    years_logLossList = []
    
    pixels_years_recall_binaryList = []
    pixels_recall_binaryList = []
    years_recall_binaryList = []
    
    pixels_years_recall_MacroList = []
    pixels_recall_MacroList = []
    years_recall_MacroList = []
    
    pixels_years_recall_MicroList = []
    pixels_recall_MicroList = []
    years_recall_MicroList = []
    
    pixels_years_jaccard_binaryList = []
    pixels_jaccard_binaryList = []
    years_jaccard_binaryList = []
    
    pixels_years_jaccard_MacroList = []
    pixels_jaccard_MacroList = []
    years_jaccard_MacroList = []
    
    pixels_years_jaccard_MicroList = []
    pixels_jaccard_MicroList = []
    years_jaccard_MicroList = []
    
    pixels_years_roc_auc_MacroList = []
    pixels_roc_auc_MacroList = []
    years_roc_auc_MacroList = []
    
    pixels_years_roc_auc_MicroList = []
    pixels_roc_auc_MicroList = []
    years_roc_auc_MicroList = []
    
    average_precisionList = []

    
    
  #used to create a list of lists of years that are excluded within each model run
    excluded_Years = []


     
    selectedParams = params
    
    for x in pixel_testVals:


        for y in year_testVals:
            trainData_X = combined_Data[combined_Data[groupVars[0]] != x]
            trainData_X = trainData_X[trainData_X[groupVars[1]] != y]
            trainData_X = trainData_X.loc[:, DataFields]


            trainData_y = target_Data[target_Data[groupVars[0]] != x]
            trainData_y = trainData_y[trainData_y[groupVars[1]] != y]


            testData_X_pixels_years = combined_Data[combined_Data[groupVars[0]] == x]
            testData_X_pixels_years = testData_X_pixels_years[testData_X_pixels_years[groupVars[1]] == y]
            testData_X_pixels_years = testData_X_pixels_years.loc[:, DataFields]

            testData_X_pixels = combined_Data[combined_Data[groupVars[0]] == x]
            testData_X_pixels = testData_X_pixels[testData_X_pixels[groupVars[1]] != y]
            testData_X_pixels = testData_X_pixels.loc[:, DataFields]

            testData_X_years = combined_Data[combined_Data[groupVars[0]] != x]
            testData_X_years = testData_X_years[testData_X_years[groupVars[1]] == y]
            testData_X_years = testData_X_years.loc[:, DataFields]



            testData_y_pixels_years = target_Data[target_Data[groupVars[0]] == x]
            testData_y_pixels_years = testData_y_pixels_years[testData_y_pixels_years[groupVars[1]] == y]


            testData_y_pixels = target_Data[target_Data[groupVars[0]] == x]
            testData_y_pixels = testData_y_pixels[testData_y_pixels[groupVars[1]] != y]


            testData_y_years = target_Data[target_Data[groupVars[0]] != x]
            testData_y_years = testData_y_years[testData_y_years[groupVars[1]] == y]
            excluded_Years.append(list(set(testData_y_years[varsToGroupBy[1]].tolist())))

            pixels_years_iterOutput = XGBoostModel_Class(trainData_X, trainData_y['value'], testData_X_pixels_years, testData_y_pixels_years['value'], selectedParams)
            pixels_iterOutput = XGBoostModel_Class(trainData_X, trainData_y['value'], testData_X_pixels, testData_y_pixels['value'], selectedParams)
            years_iterOutput = XGBoostModel_Class(trainData_X, trainData_y['value'], testData_X_years, testData_y_years['value'], selectedParams)

            
            #shap setup
            if ((x == 0) and (y ==0)):
                trainData_X_concat = copy.deepcopy(trainData_X)
                shap_values_concat = copy.deepcopy(pixels_years_iterOutput[18])
                print("initial Shap shape: ", shap_values_concat.shape)
            elif ((x >0) |(y > 0)):
                trainData_X_concat = trainData_X_concat.append(trainData_X)
                print("Shap shape: ", pixels_years_iterOutput[18].shape)
                shap_values_concat = np.vstack((shap_values_concat, pixels_years_iterOutput[18]))
            
            Models.append(pixels_years_iterOutput)


            pixels_years_MSEList.append(pixels_years_iterOutput[1])
            pixels_MSEList.append(pixels_iterOutput[1])
            years_MSEList.append(years_iterOutput[1])

            pixels_years_R2List.append(pixels_years_iterOutput[2])
            pixels_R2List.append(pixels_iterOutput[2])
            years_R2List.append(years_iterOutput[2])
            
            pixels_years_F1_binaryList.append(pixels_years_iterOutput[3])
            pixels_F1_binaryList.append(pixels_iterOutput[3])
            years_F1_binaryList.append(years_iterOutput[3])
            
            pixels_years_F1_MacroList.append(pixels_years_iterOutput[4])
            pixels_F1_MacroList.append(pixels_iterOutput[4])
            years_F1_MacroList.append(years_iterOutput[4])
            
            pixels_years_F1_MicroList.append(pixels_years_iterOutput[5])
            pixels_F1_MicroList.append(pixels_iterOutput[5])
            years_F1_MicroList.append(years_iterOutput[5])
            
            pixels_years_logLossList.append(pixels_years_iterOutput[6])
            pixels_logLossList.append(pixels_iterOutput[6])
            years_logLossList.append(years_iterOutput[6])
            
            pixels_years_recall_binaryList.append(pixels_years_iterOutput[7])
            pixels_recall_binaryList.append(pixels_iterOutput[7])
            years_recall_binaryList.append(years_iterOutput[7])
            
            pixels_years_recall_MacroList.append(pixels_years_iterOutput[8])
            pixels_recall_MacroList.append(pixels_iterOutput[8])
            years_recall_MacroList.append(years_iterOutput[8])
            
            pixels_years_recall_MicroList.append(pixels_years_iterOutput[9])
            pixels_recall_MicroList.append(pixels_iterOutput[9])
            years_recall_MicroList.append(years_iterOutput[9])
            
            pixels_years_jaccard_binaryList.append(pixels_years_iterOutput[10])
            pixels_jaccard_binaryList.append(pixels_iterOutput[10])
            years_jaccard_binaryList.append(years_iterOutput[10])
            
            pixels_years_jaccard_MacroList.append(pixels_years_iterOutput[11])
            pixels_jaccard_MacroList.append(pixels_iterOutput[11])
            years_jaccard_MacroList.append(years_iterOutput[11])
            
            pixels_years_jaccard_MicroList.append(pixels_years_iterOutput[12])
            pixels_jaccard_MicroList.append(pixels_iterOutput[12])
            years_jaccard_MicroList.append(years_iterOutput[12])
            
            pixels_years_roc_auc_MacroList.append(pixels_years_iterOutput[13])
            pixels_roc_auc_MacroList.append(pixels_iterOutput[13])
            years_roc_auc_MacroList.append(years_iterOutput[13])
            
            pixels_years_roc_auc_MicroList.append(pixels_years_iterOutput[14])
            pixels_roc_auc_MicroList.append(pixels_iterOutput[14])
            years_roc_auc_MicroList.append(years_iterOutput[14])
            
            average_precisionList.append(years_iterOutput[15])
            
            
            if((x==0) and (y==0)):
                    
                gainFrame = years_iterOutput[16]['gain']
                t_gainFrame = years_iterOutput[16]['t_gain']
                coverFrame = years_iterOutput[16]['cover']
                t_coverFrame = years_iterOutput[16]['t_cover']
                weightFrame = years_iterOutput[16]['weight']
            
            elif ((x!=0) | (y!=0)):
                gainFrame = gainFrame.append(years_iterOutput[16]['gain'])
                t_gainFrame = t_gainFrame.append(years_iterOutput[16]['t_gain'])
                coverFrame = coverFrame.append(years_iterOutput[16]['cover'])
                t_coverFrame = t_coverFrame.append(years_iterOutput[16]['t_cover'])
                weightFrame = weightFrame.append(years_iterOutput[16]['weight'])
                
    
    #create summary shap figures
    plt.clf()
    shap.summary_plot(shap_values_concat, trainData_X_concat,  plot_type="bar", show=False)
    f = plt.gcf()
    plt.tight_layout()
    f.savefig(outPath + "shap_summary_bar.tif", dpi = 300)
    
    plt.clf()
    shap.summary_plot(shap_values_concat, trainData_X_concat, show=False)
    f = plt.gcf()
    plt.tight_layout()
    f.savefig(outPath + "shap_summary.tif", dpi = 300)
    

    for q in range(shap_values_concat.shape[1]):
      f = plt.clf()
      iter_plotVar = "rank(" + str(q) + ")"
      shp_plt = shap.dependence_plot(iter_plotVar, shap_values_concat, trainData_X_concat)
      f = plt.gcf()
      plt.tight_layout()
      f.savefig(outPath + "shap_curve_" + str(q) + ".tif", dpi = 300)
    
        
    #combine MSE and R2 Lists into single DataFrame
    Models_Summary['Pixels_Years_MSE'] = pixels_years_MSEList
    Models_Summary['Pixels_MSE'] = pixels_MSEList
    Models_Summary['Years_MSE'] = years_MSEList

    Models_Summary['Pixels_Years_R2'] = pixels_years_R2List
    Models_Summary['Pixels_R2'] = pixels_R2List
    Models_Summary['Years_R2'] = years_R2List
    
    Models_Summary['Pixels_Years_F1_binaryList'] = pixels_years_F1_binaryList
    Models_Summary['Pixels_F1_binaryList'] = pixels_F1_binaryList
    Models_Summary['Years_F1_binaryList'] = years_F1_binaryList
    
    Models_Summary['Pixels_Years_F1_MacroList'] = pixels_years_F1_MacroList
    Models_Summary['Pixels_F1_MacroList'] = pixels_F1_MacroList
    Models_Summary['Years_F1_MacroList'] = years_F1_MacroList
    
    Models_Summary['Pixels_Years_F1_MicroList'] = pixels_years_F1_MicroList
    Models_Summary['Pixels_F1_MicroList'] = pixels_F1_MicroList
    Models_Summary['Years_F1_MicroList'] = years_F1_MicroList

    Models_Summary['Pixels_Years_log_loss'] = pixels_years_logLossList
    Models_Summary['Pixels_log_loss'] = pixels_logLossList
    Models_Summary['Years_log_loss'] = years_logLossList
    
    Models_Summary['Pixels_Years_recall_binaryList'] = pixels_years_recall_binaryList
    Models_Summary['Pixels_recall_binaryList'] = pixels_recall_binaryList
    Models_Summary['Years_recall_binaryList'] = years_recall_binaryList
    
    Models_Summary['Pixels_Years_recall_MacroList'] = pixels_years_recall_MacroList
    Models_Summary['Pixels_recall_MacroList'] = pixels_recall_MacroList
    Models_Summary['Years_recall_MacroList'] = years_recall_MacroList
    
    Models_Summary['Pixels_Years_recall_MicroList'] = pixels_years_recall_MicroList
    Models_Summary['Pixels_recall_MicroList'] = pixels_recall_MicroList
    Models_Summary['Years_recall_MicroList'] = years_recall_MicroList
    
    Models_Summary['Pixels_Years_jaccard_binaryList'] = pixels_years_jaccard_binaryList
    Models_Summary['Pixels_jaccard_binaryList'] = pixels_jaccard_binaryList
    Models_Summary['Years_jaccard_binaryList'] = years_jaccard_binaryList
    
    Models_Summary['Pixels_Years_jaccard_MacroList'] = pixels_years_jaccard_MacroList
    Models_Summary['Pixels_jaccard_MacroList'] = pixels_jaccard_MacroList
    Models_Summary['Years_jaccard_MacroList'] = years_jaccard_MacroList
    
    Models_Summary['Pixels_Years_jaccard_MicroList'] = pixels_years_jaccard_MicroList
    Models_Summary['Pixels_jaccard_MicroList'] = pixels_jaccard_MicroList
    Models_Summary['Years_jaccard_MicroList'] = years_jaccard_MicroList
    
    Models_Summary['Pixels_Years_roc_auc_MacroList'] = pixels_years_roc_auc_MacroList
    Models_Summary['Pixels_roc_auc_MacroList'] = pixels_roc_auc_MacroList
    Models_Summary['Years_roc_auc_MacroList'] = years_roc_auc_MacroList
    
    Models_Summary['Pixels_Years_roc_auc_MicroList'] = pixels_years_roc_auc_MicroList
    Models_Summary['Pixels_roc_auc_MicroList'] = pixels_roc_auc_MicroList
    Models_Summary['Years_roc_auc_MicroList'] = years_roc_auc_MicroList
    
    Models_Summary['average_precision'] = average_precisionList
    

    print("pixels_Years MSE Overall: ", sum(pixels_years_MSEList)/len(pixels_years_MSEList))
    print("pixels_Years R2 Overall: ", sum(pixels_years_R2List)/len(pixels_years_R2List))
    #print("pixels_Years R2 iterations: ", pixels_years_R2List)
    print("\n")
    print("pixels MSE Overall: ", sum(pixels_MSEList)/len(pixels_MSEList))
    print("pixels R2 Overall: ", sum(pixels_R2List)/len(pixels_R2List))
    #print("pixels R2 iterations: ", pixels_R2List)
    print("\n")
    print("years MSE Overall: ", sum(years_MSEList)/len(years_MSEList))
    print("years R2 Overall: ", sum(years_R2List)/len(years_R2List))
    #print("years R2 iterations: ", years_R2List)
    print("\n")

    pickling_on = open(outPath + "XGBoost_2dim.pickle", "wb")
    pickle.dump([combined_Data, target_Data, Models_Summary, Models, excluded_Years, selectedParams], pickling_on)
    pickling_on.close

    Models_Summary.to_csv(outPath + "Model_Summary_XGBOOST.csv")
    
    gainFrame.to_csv(outPath + "feature_gain_XGBOOST.csv")
    t_gainFrame.to_csv(outPath + "feature_t_gain_XGBOOST.csv")
    coverFrame.to_csv(outPath + "feature_cover_XGBOOST.csv")
    t_coverFrame.to_csv(outPath + "feature_t_cover_XGBOOST.csv")
    weightFrame.to_csv(outPath + "feature_weight_XGBOOST.csv")
    
    #export data for shap as csvs, in case we want interaction graphs later
    trainData_X_concat.to_csv(outPath + "shap_trainData_X_ConcatData.csv")
    df_shap_train = pd.DataFrame(shap_values_concat, columns=trainData_X.columns.values)
    df_shap_train.to_csv(outPath + "shap_XGB_train_ConcatData.csv")

    return combined_Data, target_Data, Models_Summary, Models, excluded_Years, selectedParams
 

def XGBoostReg_YearPredictor_Class(combined_Data_Training, target_Data_Training, 
                             preMasked_Data_Path, outPath, year_List, periodLen, 
                             DataFields, mask, params = None):
    '''annually predict fire risk- train model on combined_Data across all available years except year of interest
    save resulting predictions as csv and as tif to location 'outPath'
    
    :param combined_Data_Training: dataFrame including all desired explanatory factors 
            across all locations & years to be used in training model
    :param target_Data_Training: dataFrame including observed fire occurrences 
            across all locations & years to be used in training model
    :param preMasked_Data_Path: file path to location of files to use in predicting fire risk 
                    (note - these files should not have undergone Poisson disk masking)
    :param outPath: desired output location for predicted fire risk files (csv, pickle, and tif)
    :param year_List: list of years for which predictions are desired
    :param Datafields: list of explanatory factors to be intered into model
    :param mask: filepath of raster mask to be used in masking output predictions, 
            and as an example raster for choosing array shape and projections for .tif output files
    :param params: hyperparameters for XGBOOST regression (presumably developed from 2dimCrossval)
    :return:  returns a list of all models, accompanied by a list of years being predicted 
            - note - return output is equivalent to data exported as models.pickle
    '''
    
    model_List = []
    
    for iterYear in year_List:
        print(iterYear)
        combined_Data_iter_train = combined_Data_Training[combined_Data_Training['year'] != iterYear]
        combined_Data_iter_train = combined_Data_iter_train.loc[:, DataFields]
        
        target_Data_iter_train = target_Data_Training[target_Data_Training['year'] != iterYear]
        
        
        iter_Model = XGBClassifier(learning_rate = params["learning_rate"], 
                        max_features = params['max_features'],
                       min_samples_split = params['min_samples_split'],
                       min_samples_leaf = params['min_samples_leaf'])
        
        iter_Fit = iter_Model.fit(combined_Data_iter_train, target_Data_iter_train['value'].values)
        
        
        
        #seriesToRaster(predict_iter, templateRasterPath, outPath + "Pred_FireRisk_" + str(iterYear) + ".tif")

        full_X = pd.read_csv(preMasked_Data_Path + "CD_" + str(iterYear) + "_" + str(iterYear + periodLen - 1) + ".csv")
        full_X = full_X.loc[:, DataFields]

        data = iter_Fit.predict(full_X)
        
        
        
        data = pd.DataFrame(data, columns = ['PredClass_Masked'])
        #index_mask = image_to_series_simple(mask)             ###########
        #data['mask'] = index_mask
        #data['PredClass_Masked'] = data.apply(zeroMasker, axis =1)
        
        data.to_csv(outPath + "PredClass_" + str(iterYear) + ".csv")
        
        #output predicted risk as tiff
        seriesToRaster(data['PredClass_Masked'], mask, outPath + "PredClass_" + str(iterYear) + "_" + str(iterYear + periodLen - 1) + "XGBoost_Class.tif")
        
        data_risk = iter_Fit.predict_proba(full_X)
        
        # data_risk[1]represents predictedprobability  risk of fire, 
        #data_risk[2] represents probability of no fire
        data = pd.DataFrame(data_risk[:, 1], columns = ['PredRisk'])  
        index_mask = image_to_series_simple(mask)             ###########
        data['mask'] = index_mask
        data['PredRisk_Masked'] = data.apply(zeroMasker, axis =1)
        
        data.to_csv(outPath + "PredRisk_" + str(iterYear) + ".csv")
        
        #output predicted risk as tiff
        seriesToRaster(data['PredRisk_Masked'], mask, outPath + "PredRisk_" + str(iterYear) + "_" + str(iterYear + periodLen - 1) + "XGBoost_Class.tif")
        
        
        
        model_List.append([iter_Fit])
        
    pickling_on = open(outPath + "models.pickle", "wb")
    pickle.dump([model_List, year_List], pickling_on)
    pickling_on.close
        
        
    return model_List, year_List
       

def LogisticModel(X_train, y_train, X_test, y_test, string_output = False, 
                 selectedParams = {"penalty":'l2', "class_weight": 'balanced', # may also be none, or custom weights using 'dict'
                                  "solver": 'saga', "max_iter": 100, "n_jobs": -1}):
    '''
    Conduct logistic regression on training data and test predictive power against test data

    :param X_train: dataframe containing training data features
    :param y_train: dataframe containing training data responses
    :return: elastic net model, MSE, R-squared
    '''

    logReg = LogisticRegression(penalty = selectedParams["penalty"], 
                        class_weight = selectedParams['class_weight'],
                       solver = selectedParams['solver'],
                       max_iter = selectedParams['max_iter'],
                       n_jobs = selectedParams['n_jobs'])

    model = logReg.fit(X_train, y_train.values)  #must convert y_train to values to prevent an erroneous error warning
    predict_test = model.predict(X_test)
    predict_risk = model.predict_proba(X_test)
    predict_risk =  predict_risk[:, 1]

    MSE = model.score(X_test, y_test)
    R_Squared = r2_score(predict_test, y_test)
    
    Accuracy = skmetrics.accuracy_score(y_test, predict_test)
    BalancedAccuracy = skmetrics.balanced_accuracy_score(y_test, predict_test)
    f1_binary = skmetrics.f1_score(y_test, predict_test, average = 'binary')
    f1_macro = skmetrics.f1_score(y_test, predict_test, average = 'macro')
    f1_micro = skmetrics.f1_score(y_test, predict_test, average = 'micro')
    log_loss = skmetrics.log_loss(y_test, predict_test, labels = [0,1])
    recall_binary = skmetrics.recall_score(y_test, predict_test, average = 'binary')
    recall_macro = skmetrics.recall_score(y_test, predict_test, average = 'macro')
    recall_micro = skmetrics.recall_score(y_test, predict_test, average = 'micro')
    jaccard_binary = skmetrics.jaccard_score(y_test, predict_test,average = 'binary')
    jaccard_macro = skmetrics.jaccard_score(y_test, predict_test, average = 'macro')
    jaccard_micro = skmetrics.jaccard_score(y_test, predict_test, average = 'micro')
    roc_auc_macro = skmetrics.roc_auc_score(y_test, predict_risk, average = 'macro')
    roc_auc_micro = skmetrics.roc_auc_score(y_test, predict_risk, average = 'micro')
    average_precision = skmetrics.roc_auc_score(y_test, predict_risk)
    
    coef_Frame  = pd.DataFrame(model.coef_, columns = X_train.columns)
    
    if string_output == True:
      MSE = ("MSE = {}".format(MSE))
      R_Squared = ("R-Squared = {}".format(R_Squared))
    

    return logReg, MSE, R_Squared, f1_binary, f1_macro, f1_micro, log_loss, recall_binary, recall_macro, recall_micro, jaccard_binary, jaccard_macro, jaccard_micro, roc_auc_macro, roc_auc_micro, average_precision, predict_test, coef_Frame

def LogReg_2dimTest(combined_Data, target_Data, varsToGroupBy, groupVars, testGroups, 
                        DataFields, outPath, 
                        params = None):

    '''Conduct logistic regressions on the data, with k-fold cross-validation conducted independently 
        across both years and pixels. 
        Returns a variety of diagnostics of model performance (including f1 scores, recall, and average precision) 
        when predicting fire risk at 
        A) locations outside of the training dataset
        B) years outside of the training dataset
        C) locations and years outside of the training dataset

      Returns a list of objects, consisting of:
        0: Combined_Data file with testing/training groups labeled
        1: Target Data file with testing/training groups labeled
        2: summary dataFrame of MSE and R2 for each model run
            (against holdout data representing either novel locations, novel years, or both)
        3: list of elastic net models for use in predicting Fires in further locations/years
        4: list of list of years not used in model training for each run
    

    :param combined_Data: explanatory factors to be used in predicting fire risk
    :param target_Data: observed fire occurrences
    :param varsToGroupBy: list of (2) column names from combined_Data & target_Data to be used in creating randomized groups
    :param groupVars: list of (2) desired column names for the resulting randomized groups
    :param testGroups: number of distinct groups into which data sets should be divided (for each of two variables) 
    :param DataFields: list of data fields to be included for consideration in model construction
    :param outPath: path in which to place generated files
    :param params: hyperparameters to be used in model construction
    
    
    #Create randomly assigned groups of equal size by which to separate out subsets of data 
    #by years and by pixels for training and testing to (test against 
    #A) temporally alien, B) spatially alien, and C) completely alien conditions)
    '''

    combined_Data, target_Data = random.TestTrain_GroupMaker(combined_Data, target_Data, 
                                                             varsToGroupBy, 
                                                             groupVars, 
                                                             testGroups)

    #get list of group ids, since in cases where group # <10, may not begin at zero
    pixel_testVals = list(set(combined_Data[groupVars[0]].tolist()))
    year_testVals = list(set(combined_Data[groupVars[1]].tolist()))

    Models_Summary = pd.DataFrame([], columns = ['Pixels_Years_MSE', 'Pixels_MSE', 'Years_MSE', 
                                             'Pixels_Years_R2', 'Pixels_R2', 'Years_R2',
                                                'Pixels_Years_Accuracy', 'Pixels_Accuracy', 'Years_Accuracy',
                                                'Pixels_Years_BalancedAccuracy', 'Pixels_BalancedAccuracy', 'Years_BalancedAccuracy',
                                                'Pixels_Years_F1_binary', 'Pixels_F1_binary', 'Years_F1_binary',
                                                'Pixels_Years_F1_Macro', 'Pixels_F1_Macro', 'Years_F1_Macro',
                                                'Pixels_Years_F1_Micro', 'Pixels_F1_Micro', 'Years_F1_Micro',
                                                'Pixels_Years_logLoss', 'Pixels_logLoss', 'Years_logLoss',
                                                'Pixels_Years_recall_binary', 'Pixels_recall_binary', 'Years_recall_binary',
                                                'Pixels_Years_recall_Macro', 'Pixels_recall_Macro', 'Years_recall_Macro',
                                                'Pixels_Years_recall_Micro', 'Pixels_recall_Micro', 'Years_recall_Micro',
                                                'Pixels_Years_jaccard_binary', 'Pixels_jaccard_binary', 'Years_jaccard_binary',
                                                'Pixels_Years_jaccard_Macro', 'Pixels_jaccard_Macro', 'Years_jaccard_Macro',
                                                'Pixels_Years_jaccard_Micro', 'Pixels_jaccard_Micro', 'Years_jaccard_Micro',
                                                'Pixels_Years_roc_auc_Macro', 'Pixels_jaccard_roc_auc_Macro', 'Years_jaccard_roc_auc_Macro',
                                                 'Pixels_Years_roc_auc_Micro', 'Pixels_jaccard_roc_auc_Micro', 'Years_jaccard_roc_auc_Micro',
                                                'average_precision'])


    #used to create list of model runs
    Models = []
  
  #used to create data for entry as columns into summary DataFrame
    pixels_years_MSEList = []
    pixels_MSEList = []
    years_MSEList = []
    pixels_years_R2List = []
    pixels_R2List = []
    years_R2List = []
    
    pixels_years_F1_binaryList = []
    pixels_F1_binaryList = []
    years_F1_binaryList = []
    
    pixels_years_F1_MacroList = []
    pixels_F1_MacroList = []
    years_F1_MacroList = []
    
    pixels_years_F1_MicroList = []
    pixels_F1_MicroList = []
    years_F1_MicroList = []
    
    pixels_years_logLossList = []
    pixels_logLossList = []
    years_logLossList = []
    
    pixels_years_recall_binaryList = []
    pixels_recall_binaryList = []
    years_recall_binaryList = []
    
    pixels_years_recall_MacroList = []
    pixels_recall_MacroList = []
    years_recall_MacroList = []
    
    pixels_years_recall_MicroList = []
    pixels_recall_MicroList = []
    years_recall_MicroList = []
    
    pixels_years_jaccard_binaryList = []
    pixels_jaccard_binaryList = []
    years_jaccard_binaryList = []
    
    pixels_years_jaccard_MacroList = []
    pixels_jaccard_MacroList = []
    years_jaccard_MacroList = []
    
    pixels_years_jaccard_MicroList = []
    pixels_jaccard_MicroList = []
    years_jaccard_MicroList = []
    
    pixels_years_roc_auc_MacroList = []
    pixels_roc_auc_MacroList = []
    years_roc_auc_MacroList = []
    
    pixels_years_roc_auc_MicroList = []
    pixels_roc_auc_MicroList = []
    years_roc_auc_MicroList = []
    
    average_precisionList = []

    coef_Frame = pd.DataFrame([], columns = combined_Data.columns)

  #used to create a list of lists of years that are excluded within each model run
    excluded_Years = []


     
    selectedParams = params
    
    for x in pixel_testVals:


        for y in year_testVals:
            trainData_X = combined_Data[combined_Data[groupVars[0]] != x]
            trainData_X = trainData_X[trainData_X[groupVars[1]] != y]
            trainData_X = trainData_X.loc[:, DataFields]


            trainData_y = target_Data[target_Data[groupVars[0]] != x]
            trainData_y = trainData_y[trainData_y[groupVars[1]] != y]


            testData_X_pixels_years = combined_Data[combined_Data[groupVars[0]] == x]
            testData_X_pixels_years = testData_X_pixels_years[testData_X_pixels_years[groupVars[1]] == y]
            testData_X_pixels_years = testData_X_pixels_years.loc[:, DataFields]

            testData_X_pixels = combined_Data[combined_Data[groupVars[0]] == x]
            testData_X_pixels = testData_X_pixels[testData_X_pixels[groupVars[1]] != y]
            testData_X_pixels = testData_X_pixels.loc[:, DataFields]

            testData_X_years = combined_Data[combined_Data[groupVars[0]] != x]
            testData_X_years = testData_X_years[testData_X_years[groupVars[1]] == y]
            testData_X_years = testData_X_years.loc[:, DataFields]



            testData_y_pixels_years = target_Data[target_Data[groupVars[0]] == x]
            testData_y_pixels_years = testData_y_pixels_years[testData_y_pixels_years[groupVars[1]] == y]


            testData_y_pixels = target_Data[target_Data[groupVars[0]] == x]
            testData_y_pixels = testData_y_pixels[testData_y_pixels[groupVars[1]] != y]


            testData_y_years = target_Data[target_Data[groupVars[0]] != x]
            testData_y_years = testData_y_years[testData_y_years[groupVars[1]] == y]
            excluded_Years.append(list(set(testData_y_years[varsToGroupBy[1]].tolist())))

            pixels_years_iterOutput = LogisticModel(trainData_X, trainData_y['value'], testData_X_pixels_years, testData_y_pixels_years['value'], selectedParams)
            pixels_iterOutput = LogisticModel(trainData_X, trainData_y['value'], testData_X_pixels, testData_y_pixels['value'], selectedParams)
            years_iterOutput = LogisticModel(trainData_X, trainData_y['value'], testData_X_years, testData_y_years['value'], selectedParams)


            Models.append(pixels_years_iterOutput)


            pixels_years_MSEList.append(pixels_years_iterOutput[1])
            pixels_MSEList.append(pixels_iterOutput[1])
            years_MSEList.append(years_iterOutput[1])

            pixels_years_R2List.append(pixels_years_iterOutput[2])
            pixels_R2List.append(pixels_iterOutput[2])
            years_R2List.append(years_iterOutput[2])
            
            pixels_years_F1_binaryList.append(pixels_years_iterOutput[3])
            pixels_F1_binaryList.append(pixels_iterOutput[3])
            years_F1_binaryList.append(years_iterOutput[3])
            
            pixels_years_F1_MacroList.append(pixels_years_iterOutput[4])
            pixels_F1_MacroList.append(pixels_iterOutput[4])
            years_F1_MacroList.append(years_iterOutput[4])
            
            pixels_years_F1_MicroList.append(pixels_years_iterOutput[5])
            pixels_F1_MicroList.append(pixels_iterOutput[5])
            years_F1_MicroList.append(years_iterOutput[5])
            
            pixels_years_logLossList.append(pixels_years_iterOutput[6])
            pixels_logLossList.append(pixels_iterOutput[6])
            years_logLossList.append(years_iterOutput[6])
            
            pixels_years_recall_binaryList.append(pixels_years_iterOutput[7])
            pixels_recall_binaryList.append(pixels_iterOutput[7])
            years_recall_binaryList.append(years_iterOutput[7])
            
            pixels_years_recall_MacroList.append(pixels_years_iterOutput[8])
            pixels_recall_MacroList.append(pixels_iterOutput[8])
            years_recall_MacroList.append(years_iterOutput[8])
            
            pixels_years_recall_MicroList.append(pixels_years_iterOutput[9])
            pixels_recall_MicroList.append(pixels_iterOutput[9])
            years_recall_MicroList.append(years_iterOutput[9])
            
            pixels_years_jaccard_binaryList.append(pixels_years_iterOutput[10])
            pixels_jaccard_binaryList.append(pixels_iterOutput[10])
            years_jaccard_binaryList.append(years_iterOutput[10])
            
            pixels_years_jaccard_MacroList.append(pixels_years_iterOutput[11])
            pixels_jaccard_MacroList.append(pixels_iterOutput[11])
            years_jaccard_MacroList.append(years_iterOutput[11])
            
            pixels_years_jaccard_MicroList.append(pixels_years_iterOutput[12])
            pixels_jaccard_MicroList.append(pixels_iterOutput[12])
            years_jaccard_MicroList.append(years_iterOutput[12])
            
            pixels_years_roc_auc_MacroList.append(pixels_years_iterOutput[13])
            pixels_roc_auc_MacroList.append(pixels_iterOutput[13])
            years_roc_auc_MacroList.append(years_iterOutput[13])
            
            pixels_years_roc_auc_MicroList.append(pixels_years_iterOutput[14])
            pixels_roc_auc_MicroList.append(pixels_iterOutput[14])
            years_roc_auc_MicroList.append(years_iterOutput[14])

            average_precisionList.append(years_iterOutput[15])
            
            
            coef_Frame = coef_Frame.append(years_iterOutput[17])
            
        
    #combine MSE and R2 Lists into single DataFrame
    Models_Summary['Pixels_Years_MSE'] = pixels_years_MSEList
    Models_Summary['Pixels_MSE'] = pixels_MSEList
    Models_Summary['Years_MSE'] = years_MSEList

    Models_Summary['Pixels_Years_R2'] = pixels_years_R2List
    Models_Summary['Pixels_R2'] = pixels_R2List
    Models_Summary['Years_R2'] = years_R2List
    
    Models_Summary['Pixels_Years_F1_binary'] = pixels_years_F1_binaryList
    Models_Summary['Pixels_F1_binary'] = pixels_F1_binaryList
    Models_Summary['Years_F1_binary'] = years_F1_binaryList
    
    Models_Summary['Pixels_Years_F1_Macro'] = pixels_years_F1_MacroList
    Models_Summary['Pixels_F1_Macro'] = pixels_F1_MacroList
    Models_Summary['Years_F1_Macro'] = years_F1_MacroList
    
    Models_Summary['Pixels_Years_F1_Micro'] = pixels_years_F1_MicroList
    Models_Summary['Pixels_F1_Micro'] = pixels_F1_MicroList
    Models_Summary['Years_F1_Micro'] = years_F1_MicroList

    Models_Summary['Pixels_Years_log_loss'] = pixels_years_logLossList
    Models_Summary['Pixels_log_loss'] = pixels_logLossList
    Models_Summary['Years_log_loss'] = years_logLossList
    
    Models_Summary['Pixels_Years_recall_binary'] = pixels_years_recall_binaryList
    Models_Summary['Pixels_recall_binary'] = pixels_recall_binaryList
    Models_Summary['Years_recall_binary'] = years_recall_binaryList
    
    Models_Summary['Pixels_Years_recall_Macro'] = pixels_years_recall_MacroList
    Models_Summary['Pixels_recall_Macro'] = pixels_recall_MacroList
    Models_Summary['Years_recall_Macro'] = years_recall_MacroList
    
    Models_Summary['Pixels_Years_recall_Micro'] = pixels_years_recall_MicroList
    Models_Summary['Pixels_recall_Micro'] = pixels_recall_MicroList
    Models_Summary['Years_recall_Micro'] = years_recall_MicroList
    
    Models_Summary['Pixels_Years_jaccard_binary'] = pixels_years_jaccard_binaryList
    Models_Summary['Pixels_jaccard_binary'] = pixels_jaccard_binaryList
    Models_Summary['Years_jaccard_binary'] = years_jaccard_binaryList
    
    Models_Summary['Pixels_Years_jaccard_Macro'] = pixels_years_jaccard_MacroList
    Models_Summary['Pixels_jaccard_Macro'] = pixels_jaccard_MacroList
    Models_Summary['Years_jaccard_Macro'] = years_jaccard_MacroList
    
    Models_Summary['Pixels_Years_jaccard_Micro'] = pixels_years_jaccard_MicroList
    Models_Summary['Pixels_jaccard_Micro'] = pixels_jaccard_MicroList
    Models_Summary['Years_jaccard_Micro'] = years_jaccard_MicroList
    
    Models_Summary['Pixels_Years_roc_auc_Macro'] = pixels_years_roc_auc_MacroList
    Models_Summary['Pixels_roc_auc_Macro'] = pixels_roc_auc_MacroList
    Models_Summary['Years_roc_auc_Macro'] = years_roc_auc_MacroList
    
    Models_Summary['Pixels_Years_roc_auc_Micro'] = pixels_years_roc_auc_MicroList
    Models_Summary['Pixels_roc_auc_Micro'] = pixels_roc_auc_MicroList
    Models_Summary['Years_roc_auc_Micro'] = years_roc_auc_MicroList
    



    print("pixels_Years MSE Overall: ", sum(pixels_years_MSEList)/len(pixels_years_MSEList))
    print("pixels_Years R2 Overall: ", sum(pixels_years_R2List)/len(pixels_years_R2List))
    #print("pixels_Years R2 iterations: ", pixels_years_R2List)
    print("\n")
    print("pixels MSE Overall: ", sum(pixels_MSEList)/len(pixels_MSEList))
    print("pixels R2 Overall: ", sum(pixels_R2List)/len(pixels_R2List))
    #print("pixels R2 iterations: ", pixels_R2List)
    print("\n")
    print("years MSE Overall: ", sum(years_MSEList)/len(years_MSEList))
    print("years R2 Overall: ", sum(years_R2List)/len(years_R2List))
    #print("years R2 iterations: ", years_R2List)
    print("\n")

    pickling_on = open(outPath + "LogisticModel_2dim.pickle", "wb")
    pickle.dump([combined_Data, target_Data, Models_Summary, Models, excluded_Years, selectedParams], pickling_on)
    pickling_on.close

    Models_Summary.to_csv(outPath + "Model_Summary_LogisticModel.csv")
    coef_Frame.to_csv(outPath + "Model_Summary_LogisticModel_Coefs.csv")

    return combined_Data, target_Data, Models_Summary, Models, excluded_Years, selectedParams
   
    pixels_years_logLossList = []
    pixels_logLossList = []
    years_logLossList = []
    
    pixels_years_recall_binaryList = []
    pixels_recall_binaryList = []
    years_recall_binaryList = []
    
    pixels_years_recall_MacroList = []
    pixels_recall_MacroList = []
    years_recall_MacroList = []
    
    pixels_years_recall_MicroList = []
    pixels_recall_MicroList = []
    years_recall_MicroList = []
    
    pixels_years_jaccard_binaryList = []
    pixels_jaccard_binaryList = []
    years_jaccard_binaryList = []
    
    pixels_years_jaccard_MacroList = []
    pixels_jaccard_MacroList = []
    years_jaccard_MacroList = []
    
    pixels_years_jaccard_MicroList = []
    pixels_jaccard_MicroList = []
    years_jaccard_MicroList = []
    
    pixels_years_roc_auc_MacroList = []
    pixels_roc_auc_MacroList = []
    years_roc_auc_MacroList = []
    
    pixels_years_roc_auc_MicroList = []
    pixels_roc_auc_MicroList = []
    years_roc_auc_MicroList = []
    
    average_precisionList = []

    

  #used to create a list of lists of years that are excluded within each model run
    excluded_Years = []


     
    selectedParams = params
    
    for x in pixel_testVals:


        for y in year_testVals:
            trainData_X = combined_Data[combined_Data[groupVars[0]] != x]
            trainData_X = trainData_X[trainData_X[groupVars[1]] != y]
            trainData_X = trainData_X.loc[:, DataFields]


            trainData_y = target_Data[target_Data[groupVars[0]] != x]
            trainData_y = trainData_y[trainData_y[groupVars[1]] != y]


            testData_X_pixels_years = combined_Data[combined_Data[groupVars[0]] == x]
            testData_X_pixels_years = testData_X_pixels_years[testData_X_pixels_years[groupVars[1]] == y]
            testData_X_pixels_years = testData_X_pixels_years.loc[:, DataFields]

            testData_X_pixels = combined_Data[combined_Data[groupVars[0]] == x]
            testData_X_pixels = testData_X_pixels[testData_X_pixels[groupVars[1]] != y]
            testData_X_pixels = testData_X_pixels.loc[:, DataFields]

            testData_X_years = combined_Data[combined_Data[groupVars[0]] != x]
            testData_X_years = testData_X_years[testData_X_years[groupVars[1]] == y]
            testData_X_years = testData_X_years.loc[:, DataFields]



            testData_y_pixels_years = target_Data[target_Data[groupVars[0]] == x]
            testData_y_pixels_years = testData_y_pixels_years[testData_y_pixels_years[groupVars[1]] == y]


            testData_y_pixels = target_Data[target_Data[groupVars[0]] == x]
            testData_y_pixels = testData_y_pixels[testData_y_pixels[groupVars[1]] != y]


            testData_y_years = target_Data[target_Data[groupVars[0]] != x]
            testData_y_years = testData_y_years[testData_y_years[groupVars[1]] == y]
            excluded_Years.append(list(set(testData_y_years[varsToGroupBy[1]].tolist())))

            pixels_years_iterOutput = LogisticModel(trainData_X, trainData_y['value'], testData_X_pixels_years, testData_y_pixels_years['value'], selectedParams)
            pixels_iterOutput = LogisticModel(trainData_X, trainData_y['value'], testData_X_pixels, testData_y_pixels['value'], selectedParams)
            years_iterOutput = LogisticModel(trainData_X, trainData_y['value'], testData_X_years, testData_y_years['value'], selectedParams)


            Models.append(pixels_years_iterOutput)


            pixels_years_MSEList.append(pixels_years_iterOutput[1])
            pixels_MSEList.append(pixels_iterOutput[1])
            years_MSEList.append(years_iterOutput[1])

            pixels_years_R2List.append(pixels_years_iterOutput[2])
            pixels_R2List.append(pixels_iterOutput[2])
            years_R2List.append(years_iterOutput[2])
            
            pixels_years_F1_binaryList.append(pixels_years_iterOutput[3])
            pixels_F1_binaryList.append(pixels_iterOutput[3])
            years_F1_binaryList.append(years_iterOutput[3])
            
            pixels_years_F1_MacroList.append(pixels_years_iterOutput[4])
            pixels_F1_MacroList.append(pixels_iterOutput[4])
            years_F1_MacroList.append(years_iterOutput[4])
            
            pixels_years_F1_MicroList.append(pixels_years_iterOutput[5])
            pixels_F1_MicroList.append(pixels_iterOutput[5])
            years_F1_MicroList.append(years_iterOutput[5])
            
            pixels_years_logLossList.append(pixels_years_iterOutput[6])
            pixels_logLossList.append(pixels_iterOutput[6])
            years_logLossList.append(years_iterOutput[6])
            
            pixels_years_recall_binaryList.append(pixels_years_iterOutput[7])
            pixels_recall_binaryList.append(pixels_iterOutput[7])
            years_recall_binaryList.append(years_iterOutput[7])
            
            pixels_years_recall_MacroList.append(pixels_years_iterOutput[8])
            pixels_recall_MacroList.append(pixels_iterOutput[8])
            years_recall_MacroList.append(years_iterOutput[8])
            
            pixels_years_recall_MicroList.append(pixels_years_iterOutput[9])
            pixels_recall_MicroList.append(pixels_iterOutput[9])
            years_recall_MicroList.append(years_iterOutput[9])
            
            pixels_years_jaccard_binaryList.append(pixels_years_iterOutput[10])
            pixels_jaccard_binaryList.append(pixels_iterOutput[10])
            years_jaccard_binaryList.append(years_iterOutput[10])
            
            pixels_years_jaccard_MacroList.append(pixels_years_iterOutput[10])
            pixels_jaccard_MacroList.append(pixels_iterOutput[10])
            years_jaccard_MacroList.append(years_iterOutput[10])
            
            pixels_years_jaccard_MicroList.append(pixels_years_iterOutput[10])
            pixels_jaccard_MicroList.append(pixels_iterOutput[10])
            years_jaccard_MicroList.append(years_iterOutput[10])
            
            pixels_years_roc_auc_MacroList.append(pixels_years_iterOutput[11])
            pixels_roc_auc_MacroList.append(pixels_iterOutput[11])
            years_roc_auc_MacroList.append(years_iterOutput[11])
            
            pixels_years_roc_auc_MicroList.append(pixels_years_iterOutput[11])
            pixels_roc_auc_MicroList.append(pixels_iterOutput[11])
            years_roc_auc_MicroList.append(years_iterOutput[11])

            average_precisionList.append(years_iterOutput[12])
            
        
    #combine MSE and R2 Lists into single DataFrame
    Models_Summary['Pixels_Years_MSE'] = pixels_years_MSEList
    Models_Summary['Pixels_MSE'] = pixels_MSEList
    Models_Summary['Years_MSE'] = years_MSEList

    Models_Summary['Pixels_Years_R2'] = pixels_years_R2List
    Models_Summary['Pixels_R2'] = pixels_R2List
    Models_Summary['Years_R2'] = years_R2List
    
    Models_Summary['Pixels_Years_F1_binary'] = pixels_years_F1_binaryList
    Models_Summary['Pixels_F1_binary'] = pixels_F1_binaryList
    Models_Summary['Years_F1_binary'] = years_F1_binaryList
    
    Models_Summary['Pixels_Years_F1_Macro'] = pixels_years_F1_MacroList
    Models_Summary['Pixels_F1_Macro'] = pixels_F1_MacroList
    Models_Summary['Years_F1_Macro'] = years_F1_MacroList
    
    Models_Summary['Pixels_Years_F1_Micro'] = pixels_years_F1_MicroList
    Models_Summary['Pixels_F1_Micro'] = pixels_F1_MicroList
    Models_Summary['Years_F1_Micro'] = years_F1_MicroList

    Models_Summary['Pixels_Years_log_loss'] = pixels_years_logLossList
    Models_Summary['Pixels_log_loss'] = pixels_logLossList
    Models_Summary['Years_log_loss'] = years_logLossList
    
    Models_Summary['Pixels_Years_recall_binary'] = pixels_years_recall_binaryList
    Models_Summary['Pixels_recall_binary'] = pixels_recall_binaryList
    Models_Summary['Years_recall_binary'] = years_recall_binaryList
    
    Models_Summary['Pixels_Years_recall_Macro'] = pixels_years_recall_MacroList
    Models_Summary['Pixels_recall_Macro'] = pixels_recall_MacroList
    Models_Summary['Years_recall_Macro'] = years_recall_MacroList
    
    Models_Summary['Pixels_Years_recall_Micro'] = pixels_years_recall_MicroList
    Models_Summary['Pixels_recall_Micro'] = pixels_recall_MicroList
    Models_Summary['Years_recall_Micro'] = years_recall_MicroList
    
    Models_Summary['Pixels_Years_jaccard_binary'] = pixels_years_jaccard_binaryList
    Models_Summary['Pixels_jaccard_binary'] = pixels_jaccard_binaryList
    Models_Summary['Years_jaccard_binary'] = years_jaccard_binaryList
    
    Models_Summary['Pixels_Years_jaccard_Macro'] = pixels_years_jaccard_MacroList
    Models_Summary['Pixels_jaccard_Macro'] = pixels_jaccard_MacroList
    Models_Summary['Years_jaccard_Macro'] = years_jaccard_MacroList
    
    Models_Summary['Pixels_Years_jaccard_Micro'] = pixels_years_jaccard_MicroList
    Models_Summary['Pixels_jaccard_Micro'] = pixels_jaccard_MicroList
    Models_Summary['Years_jaccard_Micro'] = years_jaccard_MicroList
    
    Models_Summary['Pixels_Years_roc_auc_Macro'] = pixels_years_roc_auc_MacroList
    Models_Summary['Pixels_roc_auc_Macro'] = pixels_roc_auc_MacroList
    Models_Summary['Years_roc_auc_Macro'] = years_roc_auc_MacroList
    
    Models_Summary['Pixels_Years_roc_auc_Micro'] = pixels_years_roc_auc_MicroList
    Models_Summary['Pixels_roc_auc_Micro'] = pixels_roc_auc_MicroList
    Models_Summary['Years_roc_auc_Micro'] = years_roc_auc_MicroList
    



    print("pixels_Years MSE Overall: ", sum(pixels_years_MSEList)/len(pixels_years_MSEList))
    print("pixels_Years R2 Overall: ", sum(pixels_years_R2List)/len(pixels_years_R2List))
    #print("pixels_Years R2 iterations: ", pixels_years_R2List)
    print("\n")
    print("pixels MSE Overall: ", sum(pixels_MSEList)/len(pixels_MSEList))
    print("pixels R2 Overall: ", sum(pixels_R2List)/len(pixels_R2List))
    #print("pixels R2 iterations: ", pixels_R2List)
    print("\n")
    print("years MSE Overall: ", sum(years_MSEList)/len(years_MSEList))
    print("years R2 Overall: ", sum(years_R2List)/len(years_R2List))
    #print("years R2 iterations: ", years_R2List)
    print("\n")

    pickling_on = open(outPath + "LogisticModel_2dim.pickle", "wb")
    pickle.dump([combined_Data, target_Data, Models_Summary, Models, excluded_Years, selectedParams], pickling_on)
    pickling_on.close

    Models_Summary.to_csv(outPath + "Model_Summary_LogisticModel.csv")

    return combined_Data, target_Data, Models_Summary, Models, excluded_Years, selectedParams
 

def LogReg_YearPredictor(combined_Data_Training, target_Data_Training, 
                             preMasked_Data_Path, outPath, year_List, periodLen, 
                             DataFields, mask, params = None):
    '''annually predict fire risk- train model on combined_Data across all available years except year of interest
    save resulting predictions as csv and as tif to location 'outPath'
    
    :param combined_Data_Training: dataFrame including all desired explanatory factors 
            across all locations & years to be used in training model
    :param target_Data_Training: dataFrame including observed fire occurrences 
            across all locations & years to be used in training model
    :param preMasked_Data_Path: file path to location of files to use in predicting fire risk 
                    (note - these files should not have undergone Poisson disk masking)
    :param outPath: desired output location for predicted fire risk files (csv, pickle, and tif)
    :param year_List: list of years for which predictions are desired
    :param Datafields: list of explanatory factors to be intered into model
    :param mask: filepath of raster mask to be used in masking output predictions, 
            and as an example raster for choosing array shape and projections for .tif output files
    :param params: parameters for logistic regression
    :return:  returns a list of all models, accompanied by a list of years being predicted 
            - note - return output is equivalent to data exported as models.pickle
    '''
    
    model_List = []
    
    for iterYear in year_List:
        print(iterYear)
        combined_Data_iter_train = combined_Data_Training[combined_Data_Training['year'] != iterYear]
        combined_Data_iter_train = combined_Data_iter_train.loc[:, DataFields]
        
        target_Data_iter_train = target_Data_Training[target_Data_Training['year'] != iterYear]
        
        
        iter_Model = LogisticRegression(penalty = params['penalty'],
                                       class_weight = params['class_weight'],
                                       solver = params['solver'],
                                       max_iter = params['max_iter'],
                                       n_jobs = params['n_jobs'])
        
        
        
        
        
        
        
        iter_Fit = iter_Model.fit(combined_Data_iter_train, target_Data_iter_train['value'].values)
        
        
        
        #seriesToRaster(predict_iter, templateRasterPath, outPath + "Pred_FireRisk_" + str(iterYear) + ".tif")

        full_X = pd.read_csv(preMasked_Data_Path + "CD_" + str(iterYear) + "_" + str(iterYear + periodLen - 1) + ".csv")
        full_X = full_X.loc[:, DataFields]

        data = iter_Fit.predict(full_X)
        
        
        
        data = pd.DataFrame(data, columns = ['PredClass_Masked'])
        #index_mask = image_to_series_simple(mask)             ###########
        #data['mask'] = index_mask
        #data['PredClass_Masked'] = data.apply(zeroMasker, axis =1)
        
        data.to_csv(outPath + "PredClass_" + str(iterYear) + ".csv")
        
        #output predicted risk as tiff
        seriesToRaster(data['PredClass_Masked'], mask, outPath + "LogReg_PredClass_" + str(iterYear) + "_" + str(iterYear + periodLen - 1) + ".tif")
        
        data_risk = iter_Fit.predict_proba(full_X)
        
        # data_risk[1]represents predictedprobability  risk of fire, 
        #data_risk[2] represents probability of no fire
        data = pd.DataFrame(data_risk[:, 1], columns = ['PredRisk'])  
        index_mask = image_to_series_simple(mask)             ###########
        data['mask'] = index_mask
        data['PredRisk_Masked'] = data.apply(zeroMasker, axis =1)
        
        data.to_csv(outPath + "LogReg_PredRisk_" + str(iterYear) + ".csv")
        
        #output predicted risk as tiff
        seriesToRaster(data['PredRisk_Masked'], mask, outPath + "LogReg_PredRisk_" + str(iterYear) + "_" + str(iterYear + periodLen - 1) + ".tif")
        
        
        
        model_List.append([iter_Fit])
        
    pickling_on = open(outPath + "logRegModels.pickle", "wb")
    pickle.dump([model_List, year_List], pickling_on)
    pickling_on.close
        
        
    return model_List, year_List


def logGAM(X_train, y_train, X_test, y_test, 
            max_smoothing =  100, penalty_space = [0.01, 5, 100]):
    '''
    Conduct elastic net regression on training data and test predictive power against test data

    :param X_train: dataframe containing training data features
    :param y_train: dataframe containing training data responses
    :param X_test: dataframe containing training data features for testing
    :param y_test: dataframe containing training data responses for testing
    :param max_smoothing: max amount of smoothing to be allowed in the model
    :param penalty_space: space through which to search for optimal penaliztion 
        - list consisting of min penalization value, max penalization value, and number of points within that space to test
    :return: elastic net model, MSE, R-squared
    '''
    print(type(X_train))
    penalty_space = np.random.uniform(penalty_space[0], penalty_space[1], (penalty_space[2], X_train.shape[1]))
    
    gam = LogisticGAM(n_splines = max_smoothing).gridsearch(X_train, 
                                                            y_train, 
                                                            lam = penalty_space)
    model = gam.fit(X_train, y_train.values)  #must convert y_train to values to prevent an erroneous error warning
    predict_test = model.predict(X_test)
    predict_risk = model.predict_proba(X_test)

    
    
    MSE = 'null'
    
    Expl_Deviance = model.statistics_['pseudo_r2']['explained_deviance']
    McFadden_r2 = model.statistics_['pseudo_r2']['McFadden']
    McFadden_r2_adj = model.statistics_['pseudo_r2']['McFadden_adj']
    
    
    
    Accuracy = skmetrics.accuracy_score(y_test, predict_test)
    BalancedAccuracy = skmetrics.balanced_accuracy_score(y_test, predict_test)
    f1_binary = skmetrics.f1_score(y_test, predict_test, average = 'binary')
    f1_macro = skmetrics.f1_score(y_test, predict_test, average = 'macro')
    f1_micro = skmetrics.f1_score(y_test, predict_test, average = 'micro')
    log_loss = skmetrics.log_loss(y_test, predict_test, labels = [0,1])
    recall_binary = skmetrics.recall_score(y_test, predict_test, average = 'binary')
    recall_macro = skmetrics.recall_score(y_test, predict_test, average = 'macro')
    recall_micro = skmetrics.recall_score(y_test, predict_test, average = 'micro')
    jaccard_binary = skmetrics.jaccard_score(y_test, predict_test,average = 'binary')
    jaccard_macro = skmetrics.jaccard_score(y_test, predict_test, average = 'macro')
    jaccard_micro = skmetrics.jaccard_score(y_test, predict_test, average = 'micro')
    roc_auc_macro = skmetrics.roc_auc_score(y_test, predict_risk, average = 'macro')
    roc_auc_micro = skmetrics.roc_auc_score(y_test, predict_risk, average = 'micro')
    average_precision = skmetrics.roc_auc_score(y_test, predict_risk)
    
    
    output_dict = {"model": model, 
                   "McFadden_r2":McFadden_r2, "McFadden_r2_adj": McFadden_r2_adj,
                   "BalancedAccuracy": BalancedAccuracy,
                   "f1_binary": f1_binary, 
                   "f1_macro": f1_macro, "f1_micro": f1_micro,
                   "log_loss": log_loss, 
                   "recall_binary": recall_binary, "recall_macro": recall_macro, "recall_micro": recall_micro,
                   "jaccard_binary":jaccard_binary, "jaccard_macro": jaccard_macro, "jaccard_micro": jaccard_micro,
                   "roc_auc_macro": roc_auc_macro, "roc_auc_micro": roc_auc_micro, 
                   "average_precision": average_precision,
                   "predict_test": predict_test}

    
   
    
    
    return output_dict

    
def logGAM_2dimTest(combined_Data, target_Data, varsToGroupBy, groupVars, testGroups, 
                        DataFields, outPath,
                        max_smoothing = 50,
                        penalty_space = [0.01, 10, 4], # list for creating space for identifing optimal wifggliness penalization:
                        #where first value indicates minimum penalty, second indicates max penalty, and 3rd value indicates number of values
                        cv = 10):
    '''Conduct logistic regressions on the data, with k-fold cross-validation conducted independently 
        across both years and pixels. 
        Returns a variety of diagnostics of model performance (including f1 scores, recall, and average precision) 
        when predicting fire risk at 
        A) locations outside of the training dataset
        B) years outside of the training dataset
        C) locations and years outside of the training dataset

      Returns a list of objects, consisting of:
        0: Combined_Data file with testing/training groups labeled
        1: Target Data file with testing/training groups labeled
        2: summary dataFrame of MSE and R2 for each model run
            (against holdout data representing either novel locations, novel years, or both)
        3: list of elastic net models for use in predicting Fires in further locations/years
        4: list of list of years not used in model training for each run
    :param combined_Data: dataFrame including all desired explanatory factors 
            across all locations & years to be used in training model
    :param target_Data: dataFrame including observed fire occurrences 
            across all locations & years to be used in training model
    :param varsToGroupBy: list of (2) column names from combined_Data & target_Data to be used in creating randomized groups
    :param groupVars: list of (2) desired column names for the resulting randomized groups
    :param testGroups: number of distinct groups into which data sets should be divided (for each of two variables) 
    :param Datafields: list of explanatory factors to be intered into model
    :param outPath: desired output location for predicted fire risk files (csv, pickle, and tif)
    :param max_smoothing: maximum amount of smoothing to conduct in GAMs
    :param penalty_space: space to scan for amount of L2 penalization terms
        -list consisting of min penalization, max penalization, number of points within that range to test
    :return:  returns a list of all models, accompanied by a list of years being predicted 
            - note - return output is equivalent to data exported as models.pickle
    '''
    
    
    combined_Data, target_Data = random.TestTrain_GroupMaker(combined_Data, target_Data, 
                                                             varsToGroupBy, 
                                                             groupVars, 
                                                             testGroups)

    #get list of group ids, since in cases where group # <10, may not begin at zero
    pixel_testVals = list(set(combined_Data[groupVars[0]].tolist()))
    year_testVals = list(set(combined_Data[groupVars[1]].tolist()))

    Models_Summary = pd.DataFrame([], columns = [])

    #used to create list of model runs
    Models = []
    
    
  
  #used to create data for entry as columns into summary DataFrame
    pixels_years_McFaddenList = []
    pixels_McFaddenList = []
    years_McFaddenList = []
    pixels_years_McFadden_adjList = []
    pixels_McFadden_adjList = []
    years_McFadden_adjList = []
    
    
    pixels_years_BalancedAccuracyList = []
    pixels_BalancedAccuracyList = []
    years_BalancedAccuracyList = []
    
    pixels_years_F1_binaryList = []
    pixels_F1_binaryList = []
    years_F1_binaryList = []
    
    pixels_years_F1_MacroList = []
    pixels_F1_MacroList = []
    years_F1_MacroList = []
    
    pixels_years_F1_MicroList = []
    pixels_F1_MicroList = []
    years_F1_MicroList = []
    
    pixels_years_logLossList = []
    pixels_logLossList = []
    years_logLossList = []
    
    pixels_years_recall_binaryList = []
    pixels_recall_binaryList = []
    years_recall_binaryList = []
    
    pixels_years_recall_MacroList = []
    pixels_recall_MacroList = []
    years_recall_MacroList = []
    
    pixels_years_recall_MicroList = []
    pixels_recall_MicroList = []
    years_recall_MicroList = []
    
    pixels_years_jaccard_binaryList = []
    pixels_jaccard_binaryList = []
    years_jaccard_binaryList = []
    
    pixels_years_jaccard_MacroList = []
    pixels_jaccard_MacroList = []
    years_jaccard_MacroList = []
    
    pixels_years_jaccard_MicroList = []
    pixels_jaccard_MicroList = []
    years_jaccard_MicroList = []
    
    pixels_years_roc_auc_MacroList = []
    pixels_roc_auc_MacroList = []
    years_roc_auc_MacroList = []
    
    pixels_years_roc_auc_MicroList = []
    pixels_roc_auc_MicroList = []
    years_roc_auc_MicroList = []
    
    average_precisionList = []

    
    
  #used to create a list of lists of years that are excluded within each model run
    excluded_Years = []

    
    for x in pixel_testVals:


        for y in year_testVals:
            trainData_X = combined_Data[combined_Data[groupVars[0]] != x]
            trainData_X = trainData_X[trainData_X[groupVars[1]] != y]
            trainData_X = trainData_X.loc[:, DataFields]


            trainData_y = target_Data[target_Data[groupVars[0]] != x]
            trainData_y = trainData_y[trainData_y[groupVars[1]] != y]


            testData_X_pixels_years = combined_Data[combined_Data[groupVars[0]] == x]
            testData_X_pixels_years = testData_X_pixels_years[testData_X_pixels_years[groupVars[1]] == y]
            testData_X_pixels_years = testData_X_pixels_years.loc[:, DataFields]

            testData_X_pixels = combined_Data[combined_Data[groupVars[0]] == x]
            testData_X_pixels = testData_X_pixels[testData_X_pixels[groupVars[1]] != y]
            testData_X_pixels = testData_X_pixels.loc[:, DataFields]

            testData_X_years = combined_Data[combined_Data[groupVars[0]] != x]
            testData_X_years = testData_X_years[testData_X_years[groupVars[1]] == y]
            testData_X_years = testData_X_years.loc[:, DataFields]



            testData_y_pixels_years = target_Data[target_Data[groupVars[0]] == x]
            testData_y_pixels_years = testData_y_pixels_years[testData_y_pixels_years[groupVars[1]] == y]


            testData_y_pixels = target_Data[target_Data[groupVars[0]] == x]
            testData_y_pixels = testData_y_pixels[testData_y_pixels[groupVars[1]] != y]


            testData_y_years = target_Data[target_Data[groupVars[0]] != x]
            testData_y_years = testData_y_years[testData_y_years[groupVars[1]] == y]
            excluded_Years.append(list(set(testData_y_years[varsToGroupBy[1]].tolist())))
            
           
            pixels_years_iterOutput = logGAM(trainData_X.values, trainData_y['value'], testData_X_pixels_years.values, testData_y_pixels_years['value'], max_smoothing= max_smoothing, penalty_space = penalty_space)
            pixels_iterOutput = logGAM(trainData_X.values, trainData_y['value'], testData_X_pixels.values, testData_y_pixels['value'], max_smoothing=max_smoothing, penalty_space = penalty_space)
            years_iterOutput =logGAM(trainData_X.values, trainData_y['value'], testData_X_years.values, testData_y_years['value'],max_smoothing= max_smoothing, penalty_space =penalty_space)

            
    
            
            Models.append(pixels_years_iterOutput)


            pixels_years_McFaddenList.append(pixels_years_iterOutput["McFadden_r2"])
            pixels_McFaddenList.append(pixels_iterOutput["McFadden_r2"])
            years_McFaddenList.append(years_iterOutput["McFadden_r2"])

            pixels_years_McFadden_adjList.append(pixels_years_iterOutput["McFadden_r2_adj"])
            pixels_McFadden_adjList.append(pixels_iterOutput["McFadden_r2_adj"])
            years_McFadden_adjList.append(years_iterOutput["McFadden_r2_adj"])
            
            pixels_years_BalancedAccuracyList.append(pixels_years_iterOutput["BalancedAccuracy"])
            pixels_BalancedAccuracyList.append(pixels_iterOutput["BalancedAccuracy"])
            years_BalancedAccuracyList.append(years_iterOutput["BalancedAccuracy"])
            
            pixels_years_F1_binaryList.append(pixels_years_iterOutput["f1_binary"])
            pixels_F1_binaryList.append(pixels_iterOutput["f1_binary"])
            years_F1_binaryList.append(years_iterOutput["f1_binary"])
            
            pixels_years_F1_MacroList.append(pixels_years_iterOutput["f1_macro"])
            pixels_F1_MacroList.append(pixels_iterOutput["f1_macro"])
            years_F1_MacroList.append(years_iterOutput["f1_macro"])
            
            pixels_years_F1_MicroList.append(pixels_years_iterOutput["f1_micro"])
            pixels_F1_MicroList.append(pixels_iterOutput["f1_micro"])
            years_F1_MicroList.append(years_iterOutput["f1_micro"])
            
            pixels_years_logLossList.append(pixels_years_iterOutput["log_loss"])
            pixels_logLossList.append(pixels_iterOutput["log_loss"])
            years_logLossList.append(years_iterOutput["log_loss"])
            
            pixels_years_recall_binaryList.append(pixels_years_iterOutput["recall_binary"])
            pixels_recall_binaryList.append(pixels_iterOutput["recall_binary"])
            years_recall_binaryList.append(years_iterOutput["recall_binary"])
            
            pixels_years_recall_MacroList.append(pixels_years_iterOutput["recall_macro"])
            pixels_recall_MacroList.append(pixels_iterOutput["recall_macro"])
            years_recall_MacroList.append(years_iterOutput["recall_macro"])
            
            pixels_years_recall_MicroList.append(pixels_years_iterOutput["recall_micro"])
            pixels_recall_MicroList.append(pixels_iterOutput["recall_micro"])
            years_recall_MicroList.append(years_iterOutput["recall_micro"])
            
            pixels_years_jaccard_binaryList.append(pixels_years_iterOutput["jaccard_binary"])
            pixels_jaccard_binaryList.append(pixels_iterOutput["jaccard_binary"])
            years_jaccard_binaryList.append(years_iterOutput["jaccard_binary"])
            
            pixels_years_jaccard_MacroList.append(pixels_years_iterOutput["jaccard_macro"])
            pixels_jaccard_MacroList.append(pixels_iterOutput["jaccard_macro"])
            years_jaccard_MacroList.append(years_iterOutput["jaccard_macro"])
            
            pixels_years_jaccard_MicroList.append(pixels_years_iterOutput["jaccard_micro"])
            pixels_jaccard_MicroList.append(pixels_iterOutput["jaccard_micro"])
            years_jaccard_MicroList.append(years_iterOutput["jaccard_micro"])
            
            pixels_years_roc_auc_MacroList.append(pixels_years_iterOutput["roc_auc_macro"])
            pixels_roc_auc_MacroList.append(pixels_iterOutput["roc_auc_macro"])
            years_roc_auc_MacroList.append(years_iterOutput["roc_auc_macro"])
            
            pixels_years_roc_auc_MicroList.append(pixels_years_iterOutput["roc_auc_micro"])
            pixels_roc_auc_MicroList.append(pixels_iterOutput["roc_auc_micro"])
            years_roc_auc_MicroList.append(years_iterOutput["roc_auc_micro"])
            
            average_precisionList.append(years_iterOutput["average_precision"])
            
            
                
    #create marginal response graphs - uses entire dataset
    
    penalty_space_forGAM = np.random.uniform(penalty_space[0], penalty_space[1], (penalty_space[2],len(DataFields)))
    
    gam = LogisticGAM(n_splines = max_smoothing).gridsearch(combined_Data[DataFields].values, 
                                                            target_Data['value'], 
                                                            lam =  penalty_space_forGAM)
    
    #save model summary to text file
    with open(outPath +'modelDetails.txt', 'w') as f:
        with redirect_stdout(f):
            gam.summary()
    
    for j in range(len(DataFields)):
        plt.clf()
        
        XX = gam.generate_X_grid(term=j)#create grid of uniform terms for response variable
        plt.plot(XX[:, j], gam.partial_dependence(term = j, X=XX))
        plt.plot(XX[:, j], gam.partial_dependence(term=j, X=XX, width=.95)[1], c='r', ls='--')
        plt.title(DataFields[j])
        plt.tight_layout()
        plt.savefig(outPath + "Marginal_Response_"+ DataFields[j] + ".tif")
        
        
        
    #combine MSE and R2 Lists into single DataFrame
    Models_Summary['Pixels_Years_McFadden'] = pixels_years_McFaddenList
    Models_Summary['Pixels_McFadden'] = pixels_McFaddenList
    Models_Summary['Years_McFadden'] = years_McFaddenList

    Models_Summary['Pixels_Years_McFaddenAdj'] = pixels_years_McFadden_adjList
    Models_Summary['Pixels_McFaddenAdj'] = pixels_McFadden_adjList
    Models_Summary['Years_McFaddenAdj'] = years_McFadden_adjList
    
    Models_Summary['Pixels_Years_BalancedAccuracy'] = pixels_years_BalancedAccuracyList
    Models_Summary['Pixels_BalancedAccuracy'] = pixels_BalancedAccuracyList
    Models_Summary['Years_BalancedAccuracy'] = years_BalancedAccuracyList
    
    Models_Summary['Pixels_Years_F1_binaryList'] = pixels_years_F1_binaryList
    Models_Summary['Pixels_F1_binaryList'] = pixels_F1_binaryList
    Models_Summary['Years_F1_binaryList'] = years_F1_binaryList
    
    Models_Summary['Pixels_Years_F1_MacroList'] = pixels_years_F1_MacroList
    Models_Summary['Pixels_F1_MacroList'] = pixels_F1_MacroList
    Models_Summary['Years_F1_MacroList'] = years_F1_MacroList
    
    Models_Summary['Pixels_Years_F1_MicroList'] = pixels_years_F1_MicroList
    Models_Summary['Pixels_F1_MicroList'] = pixels_F1_MicroList
    Models_Summary['Years_F1_MicroList'] = years_F1_MicroList

    Models_Summary['Pixels_Years_log_loss'] = pixels_years_logLossList
    Models_Summary['Pixels_log_loss'] = pixels_logLossList
    Models_Summary['Years_log_loss'] = years_logLossList
    
    Models_Summary['Pixels_Years_recall_binaryList'] = pixels_years_recall_binaryList
    Models_Summary['Pixels_recall_binaryList'] = pixels_recall_binaryList
    Models_Summary['Years_recall_binaryList'] = years_recall_binaryList
    
    Models_Summary['Pixels_Years_recall_MacroList'] = pixels_years_recall_MacroList
    Models_Summary['Pixels_recall_MacroList'] = pixels_recall_MacroList
    Models_Summary['Years_recall_MacroList'] = years_recall_MacroList
    
    Models_Summary['Pixels_Years_recall_MicroList'] = pixels_years_recall_MicroList
    Models_Summary['Pixels_recall_MicroList'] = pixels_recall_MicroList
    Models_Summary['Years_recall_MicroList'] = years_recall_MicroList
    
    Models_Summary['Pixels_Years_jaccard_binaryList'] = pixels_years_jaccard_binaryList
    Models_Summary['Pixels_jaccard_binaryList'] = pixels_jaccard_binaryList
    Models_Summary['Years_jaccard_binaryList'] = years_jaccard_binaryList
    
    Models_Summary['Pixels_Years_jaccard_MacroList'] = pixels_years_jaccard_MacroList
    Models_Summary['Pixels_jaccard_MacroList'] = pixels_jaccard_MacroList
    Models_Summary['Years_jaccard_MacroList'] = years_jaccard_MacroList
    
    Models_Summary['Pixels_Years_jaccard_MicroList'] = pixels_years_jaccard_MicroList
    Models_Summary['Pixels_jaccard_MicroList'] = pixels_jaccard_MicroList
    Models_Summary['Years_jaccard_MicroList'] = years_jaccard_MicroList
    
    Models_Summary['Pixels_Years_roc_auc_MacroList'] = pixels_years_roc_auc_MacroList
    Models_Summary['Pixels_roc_auc_MacroList'] = pixels_roc_auc_MacroList
    Models_Summary['Years_roc_auc_MacroList'] = years_roc_auc_MacroList
    
    Models_Summary['Pixels_Years_roc_auc_MicroList'] = pixels_years_roc_auc_MicroList
    Models_Summary['Pixels_roc_auc_MicroList'] = pixels_roc_auc_MicroList
    Models_Summary['Years_roc_auc_MicroList'] = years_roc_auc_MicroList
    
    Models_Summary['average_precision'] = average_precisionList
    

    pickling_on = open(outPath + "logGAM_2dim.pickle", "wb")
    pickle.dump([combined_Data, target_Data, Models_Summary, Models, excluded_Years], pickling_on)
    pickling_on.close

    Models_Summary.to_csv(outPath + "Model_Summary_logGAM.csv")

    return combined_Data, target_Data, Models_Summary, Models, excluded_Years


def logGAM_YearPredictor_Class(combined_Data_Training, target_Data_Training, 
                                preMasked_Data_Path, outPath, year_List, periodLen, 
                                DataFields, mask, max_smoothing = 50,
                                penalty_space = [0.01, 10, 4], # list for creating space for identifing optimal wifggliness penalization:
                                #where first value indicates minimum penalty, second indicates max penalty, and 3rd value indicates number of values
                                ):
    '''annually predict fire risk- train model on combined_Data across all available years except year of interest
    save resulting predictions as csv and as tif to location 'outPath'
    
    :param combined_Data_Training: dataFrame including all desired explanatory factors 
            across all locations & years to be used in training model
    :param target_Data_Training: dataFrame including observed fire occurrences 
            across all locations & years to be used in training model
    :param preMasked_Data_Path: file path to location of files to use in predicting fire risk 
                    (note - these files should not have undergone Poisson disk masking)
    :param outPath: desired output location for predicted fire risk files (csv, pickle, and tif)
    :param year_List: list of years for which predictions are desired
    :param Datafields: list of explanatory factors to be intered into model
    :param mask: filepath of raster mask to be used in masking output predictions, 
            and as an example raster for choosing array shape and projections for .tif output files
    :param params: hyperparameters for XGBOOST regression (presumably developed from 2dimCrossval)
    :return:  returns a list of all models, accompanied by a list of years being predicted 
            - note - return output is equivalent to data exported as models.pickle
    '''
    
    model_List = []
    
    for iterYear in year_List:
        print(iterYear)
        combined_Data_iter_train = combined_Data_Training[combined_Data_Training['year'] != iterYear]
        combined_Data_iter_train = combined_Data_iter_train.loc[:, DataFields]
        
        target_Data_iter_train = target_Data_Training[target_Data_Training['year'] != iterYear]
        
        
        penalty_space_forGAM = np.random.uniform(penalty_space[0], penalty_space[1], (penalty_space[2],len(DataFields)))
        
        iter_gam = LogisticGAM(n_splines = max_smoothing).gridsearch(combined_Data_iter_train.values, 
                                                            target_Data_iter_train['value'].values, 
                                                            lam =  penalty_space_forGAM)
        
        iter_Fit = iter_gam.fit(combined_Data_iter_train, target_Data_iter_train['value'].values)
        
        
        
        #seriesToRaster(predict_iter, templateRasterPath, outPath + "Pred_FireRisk_" + str(iterYear) + ".tif")

        full_X = pd.read_csv(preMasked_Data_Path + "CD_" + str(iterYear) + "_" + str(iterYear + periodLen - 1) + ".csv")
        full_X = full_X.loc[:, DataFields]

        data = iter_Fit.predict(full_X)
        
        
        
        data = pd.DataFrame(data, columns = ['PredClass_Masked'])
        #index_mask = image_to_series_simple(mask)             ###########
        #data['mask'] = index_mask
        #data['PredClass_Masked'] = data.apply(zeroMasker, axis =1)
        
        data.to_csv(outPath + "PredClass_" + str(iterYear) + ".csv")
        
        #output predicted risk as tiff
        seriesToRaster(data['PredClass_Masked'], mask, outPath + "PredClass_" + str(iterYear) + "_" + str(iterYear + periodLen - 1) + "LogGam_Class.tif")
        
        data_risk = iter_Fit.predict_proba(full_X)
        print(data_risk.shape)
        # data_risk[1]represents predictedprobability  risk of fire, 
        #data_risk[2] represents probability of no fire
        data = pd.DataFrame(data_risk, columns = ['PredRisk'])  
        index_mask = image_to_series_simple(mask)             ###########
        data['mask'] = index_mask
        data['PredRisk_Masked'] = data.apply(zeroMasker, axis =1)
        
        data.to_csv(outPath + "PredRisk_" + str(iterYear) + ".csv")
        
        #output predicted risk as tiff
        seriesToRaster(data['PredRisk_Masked'], mask, outPath + "PredRisk_" + str(iterYear) + "_" + str(iterYear + periodLen - 1) + "LogGam_Class.tif")
        
        
        
        model_List.append([iter_Fit])
        
    pickling_on = open(outPath + "models.pickle", "wb")
    pickle.dump([model_List, year_List], pickling_on)
    pickling_on.close
        
        
    return model_List, year_List


def dataFrame_to_r(in_frame):
    #convert pandas dataframe to r dataframe
    base = importr('base')

    from rpy2.robjects import pandas2ri
    pandas2ri.activate()


    #Convert pandas to r
    r_in_frame = pandas2ri.py2ri(in_frame)
    

    #calling function under base package
    #print(base.summary(r_combined_Data))
    #print(type(r_combined_Data))
    #print(r_combined_Data)
    return r_in_frame


def R_GAM_YearPredictor_Class(combined_Data_Training, target_Data_Training, 
                                preMasked_Data_Path, outPath, year_List, periodLen, 
                                DataFields, mask,
                               splineType = 'cs', # list for creating space for identifing optimal wifggliness penalization:
                               familyType = "binomial" #where first value indicates minimum penalty, second indicates max penalty, and 3rd value indicates number of values
                                ):
    '''annually predict fire risk- train model on combined_Data across all available years except year of interest
    save resulting predictions as csv and as tif to location 'outPath'
    
    :param combined_Data_Training: dataFrame including all desired explanatory factors 
            across all locations & years to be used in training model
    :param target_Data_Training: dataFrame including observed fire occurrences 
            across all locations & years to be used in training model
    :param preMasked_Data_Path: file path to location of files to use in predicting fire risk 
                    (note - these files should not have undergone Poisson disk masking)
    :param outPath: desired output location for predicted fire risk files (csv, pickle, and tif)
    :param year_List: list of years for which predictions are desired
    :param Datafields: list of explanatory factors to be intered into model
    :param mask: filepath of raster mask to be used in masking output predictions, 
            and as an example raster for choosing array shape and projections for .tif output files
    :param splineType: type of splines to use - default is cs, which is cubic with shrinkage
    :param familyType: type of GAM to use: default binomial
    :return:  returns a list of all models, accompanied by a list of years being predicted 
            - note - return output is equivalent to data exported as models.pickle
    '''
    
    mgcv = importr('mgcv') # import mgcv library from r
    stats = importr('stats') # import stats library from r
    base = importr('base') # import base library from r
    
    
    model_List = []
    
    
    
    #set up formla for model
    iter_formula = 'value~'
    for iterparam in DataFields:
        iter_formula += " + s(" + iterparam + ', bs = \"' + splineType + '")'
    
    
    for iterYear in year_List:
        combined_Data_iter_train = combined_Data_Training[combined_Data_Training['year'] != iterYear]
        combined_Data_iter_train = combined_Data_iter_train.loc[:, DataFields]
        
        target_Data_iter_train = target_Data_Training[target_Data_Training['year'] != iterYear]
        combined_Data_iter_train['value'] = target_Data_iter_train['value']
        
        
        r_combined_Data_iter_train = dataFrame_to_r(combined_Data_iter_train)
        
        
       
        
        #run model 
        model = mgcv.gam(formula = base.eval(base.parse(text=iter_formula)),family = base.eval(base.parse(text=familyType)), data = r_combined_Data_iter_train)
        
        
        
        #seriesToRaster(predict_iter, templateRasterPath, outPath + "Pred_FireRisk_" + str(iterYear) + ".tif")

        full_X = pd.read_csv(preMasked_Data_Path + "CD_" + str(iterYear) + "_" + str(iterYear + periodLen - 1) + ".csv")
        full_X = full_X.loc[:, DataFields]
        full_X = dataFrame_to_r(full_X)

        
        predict_proba = stats.predict(model,full_X, type = 'response')
        predict_proba = np.array(predict_proba)
        


        print(predict_proba.shape)
        # data_risk[1]represents predictedprobability  risk of fire, 
        #data_risk[2] represents probability of no fire
        data = pd.DataFrame(predict_proba, columns = ['PredRisk'])  
        index_mask = image_to_series_simple(mask)             ###########
        data['mask'] = index_mask
        data['PredRisk_Masked'] = data.apply(zeroMasker, axis =1)
        
        data.to_csv(outPath + "PredRisk_" + str(iterYear) + ".csv")
        
        #output predicted risk as tiff
        seriesToRaster(data['PredRisk_Masked'], mask, outPath + "PredRisk_" + str(iterYear) + "_" + str(iterYear + periodLen - 1) + "LogGam_Class.tif")
    
        
def predict_Test(model, testData, threshold, suffix = ''):
    '''iterates predictions across pixels_years, pixels, and years 
    to conduct model diagnostics across different forms of cross-validation:
    is called by R_GAM in 2dim cross validation
    '''
    
    mgcv = importr('mgcv') # import mgcv library from r
    stats = importr('stats') # import stats library from r
    base = importr('base') # import base library from r
    
    r_testData = dataFrame_to_r(testData)
    
    predict_proba = stats.predict(model,r_testData, type = 'response')
    predict_risk = np.array(predict_proba)
    predict_test = copy.deepcopy(predict_risk)
    
    predict_test = np.where(predict_test >threshold, 1.0, predict_test)
    predict_test = np.where(predict_test <=threshold, 0.0, predict_test)
    
    
    y_test = np.array(testData['value'])
    
    r_summary = base.summary(model)
    
    r2 = np.array(r_summary.rx('r.sq')[0])[0]
    se = np.array(r_summary.rx('se')[0])[0]
    
    Accuracy = skmetrics.accuracy_score(y_test, predict_test)
    BalancedAccuracy = skmetrics.balanced_accuracy_score(y_test, predict_test)
    f1_binary = skmetrics.f1_score(y_test, predict_test, average = 'binary')
    f1_macro = skmetrics.f1_score(y_test, predict_test, average = 'macro')
    f1_micro = skmetrics.f1_score(y_test, predict_test, average = 'micro')
    log_loss = skmetrics.log_loss(y_test, predict_test, labels = [0,1])
    recall_binary = skmetrics.recall_score(y_test, predict_test, average = 'binary')
    recall_macro = skmetrics.recall_score(y_test, predict_test, average = 'macro')
    recall_micro = skmetrics.recall_score(y_test, predict_test, average = 'micro')
    jaccard_binary = skmetrics.jaccard_score(y_test, predict_test,average = 'binary')
    jaccard_macro = skmetrics.jaccard_score(y_test, predict_test, average = 'macro')
    jaccard_micro = skmetrics.jaccard_score(y_test, predict_test, average = 'micro')
    roc_auc_macro = skmetrics.roc_auc_score(y_test, predict_risk, average = 'macro')
    roc_auc_micro = skmetrics.roc_auc_score(y_test, predict_risk, average = 'micro')
    average_precision = skmetrics.roc_auc_score(y_test, predict_risk)
   

    output_dict = {"r2_"+ suffix:r2,
                   "BalancedAccuracy_"+ suffix: BalancedAccuracy,
                   "f1_binary_"+ suffix: f1_binary, 
                   "f1_macro_"+ suffix: f1_macro, "f1_micro_"+ suffix: f1_micro,
                   "log_loss_"+ suffix: log_loss, 
                   "recall_binary_"+ suffix: recall_binary, "recall_macro_"+ suffix: recall_macro, "recall_micro_"+ suffix: recall_micro,
                   "jaccard_binary_"+ suffix:jaccard_binary, "jaccard_macro_"+ suffix: jaccard_macro, "jaccard_micro_"+ suffix: jaccard_micro,
                   "roc_auc_macro_"+ suffix: roc_auc_macro, "roc_auc_micro_"+ suffix: roc_auc_micro, 
                   "average_precision_"+ suffix: average_precision}
    
    return output_dict

def R_GAM(X_train, y_train, X_test_pixels, y_test_pixels, 
          X_test_years, y_test_years, 
          X_test_pixels_years, y_test_pixels_years, 
          formula, familyType, threshold = 0.5):
    '''
    Conduct elastic net regression on training data and test predictive power against test data

    :param X_train: pandas dataframe containing training data features
    :param y_train: pandas dataframe containing training data responses
    :param X_test: pandas dataframe containing training data features for testing
    :param y_test: pandas dataframe containing training data responses for testing
    :param iterFormula: formula to be entered into r model
    :param threshold:  threshold above which prediction will be set to 1, below which it will be set to 0
         :return: dictionary of model object and diagnostic parameters
    '''
    
    mgcv = importr('mgcv') # import mgcv library from r
    stats = importr('stats') # import stats library from r
    base = importr('base') # import base library from r
    
    
    
   
    
    #combine value into X train and test data, convert to r dataFrames
    X_train['value'] = y_train['value']
    r_train = dataFrame_to_r(X_train)
    
    
    
    X_test_pixels['value'] = y_test_pixels['value']
    #r_test_pixels = dataFrame_to_r(X_test_pixels)
    
    X_test_years['value'] = y_test_years['value']
    #r_test_years = dataFrame_to_r(X_test_years)
    
    X_test_pixels_years['value'] = y_test_pixels_years['value']
    #r_test_pixels_years = dataFrame_to_r(X_test_pixels_years)
    


    model = mgcv.gam(formula = base.eval(base.parse(text=formula)),family = base.eval(base.parse(text=familyType)), data = r_train)

    blockDict = {"pixels": X_test_pixels, "years": X_test_years, "pixels_years": X_test_pixels_years}
    
    out_stats = {'blank': ''}
    for i in blockDict:
        iterData = blockDict[i]
        suffix = "_" + i
        iterDict = predict_Test(model, iterData, threshold, suffix = i)
        out_stats.update(iterDict)
    
    
    output_dict = {"model": model, 
                   "out_stats": out_stats}
    
    return output_dict

def R_logGAM_2dimTest(combined_Data, target_Data, varsToGroupBy, groupVars, testGroups, 
                        DataFields, outPath,
                        splineType = 'cs', # list for creating space for identifing optimal wifggliness penalization:
                        familyType = "binomial" #where first value indicates minimum penalty, second indicates max penalty, and 3rd value indicates number of values
                        ):
    '''Conduct logistic regressions on the data, with k-fold cross-validation conducted independently 
        across both years and pixels. 
        Returns a variety of diagnostics of model performance (including f1 scores, recall, and average precision) 
        when predicting fire risk at 
        A) locations outside of the training dataset
        B) years outside of the training dataset
        C) locations and years outside of the training dataset

      Returns a list of objects, consisting of:
        0: Combined_Data file with testing/training groups labeled
        1: Target Data file with testing/training groups labeled
        2: summary dataFrame of MSE and R2 for each model run
            (against holdout data representing either novel locations, novel years, or both)
        3: list of elastic net models for use in predicting Fires in further locations/years
        4: list of list of years not used in model training for each run
    :param combined_Data: dataFrame including all desired explanatory factors 
            across all locations & years to be used in training model
    :param target_Data: dataFrame including observed fire occurrences 
            across all locations & years to be used in training model
    :param varsToGroupBy: list of (2) column names from combined_Data & target_Data to be used in creating randomized groups
    :param groupVars: list of (2) desired column names for the resulting randomized groups
    :param testGroups: number of distinct groups into which data sets should be divided (for each of two variables) 
    :param Datafields: list of explanatory factors to be intered into model
    :param outPath: desired output location for predicted fire risk files (csv, pickle, and tif)
    :param max_smoothing: maximum amount of smoothing to conduct in GAMs
    :param penalty_space: space to scan for amount of L2 penalization terms
        -list consisting of min penalization, max penalization, number of points within that range to test
    :return:  returns a list of all models, accompanied by a list of years being predicted 
            - note - return output is equivalent to data exported as models.pickle
    '''
    
    
    
    
    combined_Data, target_Data = random.TestTrain_GroupMaker(combined_Data, target_Data, 
                                                             varsToGroupBy, 
                                                             groupVars, 
                                                             testGroups)

    #get list of group ids, since in cases where group # <10, may not begin at zero
    pixel_testVals = list(set(combined_Data[groupVars[0]].tolist()))
    year_testVals = list(set(combined_Data[groupVars[1]].tolist()))

    Models_Summary = pd.DataFrame([], columns = [])

    #used to create list of model runs
    models = []
    
    
  
  #used to create data for entry as columns into summary DataFrame
    
    
    pixels_years_r2List = []
    pixels_r2List = []
    years_r2List = []
    
    pixels_years_BalancedAccuracyList = []
    pixels_BalancedAccuracyList = []
    years_BalancedAccuracyList = []
    
    pixels_years_F1_binaryList = []
    pixels_F1_binaryList = []
    years_F1_binaryList = []
    
    pixels_years_F1_MacroList = []
    pixels_F1_MacroList = []
    years_F1_MacroList = []
    
    pixels_years_F1_MicroList = []
    pixels_F1_MicroList = []
    years_F1_MicroList = []
    
    pixels_years_logLossList = []
    pixels_logLossList = []
    years_logLossList = []
    
    pixels_years_recall_binaryList = []
    pixels_recall_binaryList = []
    years_recall_binaryList = []
    
    pixels_years_recall_MacroList = []
    pixels_recall_MacroList = []
    years_recall_MacroList = []
    
    pixels_years_recall_MicroList = []
    pixels_recall_MicroList = []
    years_recall_MicroList = []
    
    pixels_years_jaccard_binaryList = []
    pixels_jaccard_binaryList = []
    years_jaccard_binaryList = []
    
    pixels_years_jaccard_MacroList = []
    pixels_jaccard_MacroList = []
    years_jaccard_MacroList = []
    
    pixels_years_jaccard_MicroList = []
    pixels_jaccard_MicroList = []
    years_jaccard_MicroList = []
    
    pixels_years_roc_auc_MacroList = []
    pixels_roc_auc_MacroList = []
    years_roc_auc_MacroList = []
    
    pixels_years_roc_auc_MicroList = []
    pixels_roc_auc_MicroList = []
    years_roc_auc_MicroList = []
    
    pixels_years_average_precisionList = []
    pixels_average_precisionList = []
    years_average_precisionList = []

    
    
  #used to create a list of lists of years that are excluded within each model run
    excluded_Years = []

    #set up formula for use in model
    formula = 'value~'
    for iterparam in DataFields:
        formula += " + s(" + iterparam + ', bs = \"' + splineType + '")'
    
    
    for x in pixel_testVals:


        for y in year_testVals:
            trainData_X = combined_Data[combined_Data[groupVars[0]] != x]
            trainData_X = trainData_X[trainData_X[groupVars[1]] != y]
            trainData_X = trainData_X.loc[:, DataFields]


            trainData_y = target_Data[target_Data[groupVars[0]] != x]
            trainData_y = trainData_y[trainData_y[groupVars[1]] != y]


            testData_X_pixels_years = combined_Data[combined_Data[groupVars[0]] == x]
            testData_X_pixels_years = testData_X_pixels_years[testData_X_pixels_years[groupVars[1]] == y]
            testData_X_pixels_years = testData_X_pixels_years.loc[:, DataFields]

            testData_X_pixels = combined_Data[combined_Data[groupVars[0]] == x]
            testData_X_pixels = testData_X_pixels[testData_X_pixels[groupVars[1]] != y]
            testData_X_pixels = testData_X_pixels.loc[:, DataFields]

            testData_X_years = combined_Data[combined_Data[groupVars[0]] != x]
            testData_X_years = testData_X_years[testData_X_years[groupVars[1]] == y]
            testData_X_years = testData_X_years.loc[:, DataFields]



            testData_y_pixels_years = target_Data[target_Data[groupVars[0]] == x]
            testData_y_pixels_years = testData_y_pixels_years[testData_y_pixels_years[groupVars[1]] == y]


            testData_y_pixels = target_Data[target_Data[groupVars[0]] == x]
            testData_y_pixels = testData_y_pixels[testData_y_pixels[groupVars[1]] != y]


            testData_y_years = target_Data[target_Data[groupVars[0]] != x]
            testData_y_years = testData_y_years[testData_y_years[groupVars[1]] == y]
            excluded_Years.append(list(set(testData_y_years[varsToGroupBy[1]].tolist())))
            
           
        
            
        
            iterOutput = R_GAM(X_train = trainData_X, y_train = trainData_y, 
                      X_test_pixels = testData_X_pixels, y_test_pixels = testData_y_pixels, 
                      X_test_years = testData_X_years, y_test_years = testData_y_years, 
                      X_test_pixels_years = testData_X_pixels_years, y_test_pixels_years = testData_y_pixels_years, 
                      formula = formula,
                      familyType = familyType, threshold = 0.5)
            
            iter_stats = iterOutput['out_stats'] #get dictionary of stats summaries
            
            pixels_years_r2List.append(iter_stats["r2_pixels_years"])
            pixels_r2List.append(iter_stats["r2_pixels"])
            years_r2List.append(iter_stats["r2_years"])

            pixels_years_BalancedAccuracyList.append(iter_stats["BalancedAccuracy_pixels_years"])
            pixels_BalancedAccuracyList.append(iter_stats["BalancedAccuracy_pixels"])
            years_BalancedAccuracyList.append(iter_stats["BalancedAccuracy_years"])
            
            pixels_years_F1_binaryList.append(iter_stats["f1_binary_pixels_years"])
            pixels_F1_binaryList.append(iter_stats["f1_binary_pixels"])
            years_F1_binaryList.append(iter_stats["f1_binary_years"])
            
            pixels_years_F1_MacroList.append(iter_stats["f1_macro_pixels_years"])
            pixels_F1_MacroList.append(iter_stats["f1_macro_pixels"])
            years_F1_MacroList.append(iter_stats["f1_macro_years"])
            
            pixels_years_F1_MicroList.append(iter_stats["f1_micro_pixels_years"])
            pixels_F1_MicroList.append(iter_stats["f1_micro_pixels"])
            years_F1_MicroList.append(iter_stats["f1_micro_years"])
            
            pixels_years_logLossList.append(iter_stats["log_loss_pixels_years"])
            pixels_logLossList.append(iter_stats["log_loss_pixels"])
            years_logLossList.append(iter_stats["log_loss_years"])
            
            pixels_years_recall_binaryList.append(iter_stats["recall_binary_pixels_years"])
            pixels_recall_binaryList.append(iter_stats["recall_binary_pixels"])
            years_recall_binaryList.append(iter_stats["recall_binary_years"])
            
            pixels_years_recall_MacroList.append(iter_stats["recall_macro_pixels_years"])
            pixels_recall_MacroList.append(iter_stats["recall_macro_pixels"])
            years_recall_MacroList.append(iter_stats["recall_macro_years"])
            
            pixels_years_recall_MicroList.append(iter_stats["recall_micro_pixels_years"])
            pixels_recall_MicroList.append(iter_stats["recall_micro_pixels"])
            years_recall_MicroList.append(iter_stats["recall_micro_years"])
            
            pixels_years_jaccard_binaryList.append(iter_stats["jaccard_binary_pixels_years"])
            pixels_jaccard_binaryList.append(iter_stats["jaccard_binary_pixels"])
            years_jaccard_binaryList.append(iter_stats["jaccard_binary_years"])
            
            pixels_years_jaccard_MacroList.append(iter_stats["jaccard_macro_pixels_years"])
            pixels_jaccard_MacroList.append(iter_stats["jaccard_macro_pixels"])
            years_jaccard_MacroList.append(iter_stats["jaccard_macro_years"])
            
            pixels_years_jaccard_MicroList.append(iter_stats["jaccard_micro_pixels_years"])
            pixels_jaccard_MicroList.append(iter_stats["jaccard_micro_pixels"])
            years_jaccard_MicroList.append(iter_stats["jaccard_micro_years"])
            
            pixels_years_roc_auc_MacroList.append(iter_stats["roc_auc_macro_pixels_years"])
            pixels_roc_auc_MacroList.append(iter_stats["roc_auc_macro_pixels"])
            years_roc_auc_MacroList.append(iter_stats["roc_auc_macro_years"])
            
            pixels_years_roc_auc_MicroList.append(iter_stats["roc_auc_micro_pixels_years"])
            pixels_roc_auc_MicroList.append(iter_stats["roc_auc_micro_pixels"])
            years_roc_auc_MicroList.append(iter_stats["roc_auc_micro_years"])
            
            pixels_years_average_precisionList.append(iter_stats["average_precision_pixels_years"])
            pixels_average_precisionList.append(iter_stats["average_precision_pixels"])
            years_average_precisionList.append(iter_stats["average_precision_years"])
            models.append(iterOutput['model'])
            
    '''            
    #create marginal response graphs - uses entire dataset
    
    penalty_space_forGAM = np.random.uniform(penalty_space[0], penalty_space[1], (penalty_space[2],len(DataFields)))
    
    gam = LogisticGAM(n_splines = max_smoothing).gridsearch(combined_Data[DataFields].values, 
                                                            target_Data['value'], 
                                                            lam =  penalty_space_forGAM)
    
    #save model summary to text file
    with open(outPath +'modelDetails.txt', 'w') as f:
        with redirect_stdout(f):
            gam.summary()
    
    for j in range(len(DataFields)):
        plt.clf()
        
        XX = gam.generate_X_grid(term=j)#create grid of uniform terms for response variable
        plt.plot(XX[:, j], gam.partial_dependence(term = j, X=XX))
        plt.plot(XX[:, j], gam.partial_dependence(term=j, X=XX, width=.95)[1], c='r', ls='--')
        plt.title(DataFields[j])
        plt.tight_layout()
        plt.savefig(outPath + "Marginal_Response_"+ DataFields[j] + ".tif")
        
    '''
        
    #combine MSE and R2 Lists into single DataFrame
    
    Models_Summary['Pixels_Years_r2'] = pixels_years_r2List
    Models_Summary['Pixels_r2'] = pixels_r2List
    Models_Summary['Years_r2'] = years_r2List
    
    Models_Summary['Pixels_Years_BalancedAccuracy'] = pixels_years_BalancedAccuracyList
    Models_Summary['Pixels_BalancedAccuracy'] = pixels_BalancedAccuracyList
    Models_Summary['Years_BalancedAccuracy'] = years_BalancedAccuracyList
    
    Models_Summary['Pixels_Years_F1_binaryList'] = pixels_years_F1_binaryList
    Models_Summary['Pixels_F1_binaryList'] = pixels_F1_binaryList
    Models_Summary['Years_F1_binaryList'] = years_F1_binaryList
    
    Models_Summary['Pixels_Years_F1_MacroList'] = pixels_years_F1_MacroList
    Models_Summary['Pixels_F1_MacroList'] = pixels_F1_MacroList
    Models_Summary['Years_F1_MacroList'] = years_F1_MacroList
    
    Models_Summary['Pixels_Years_F1_MicroList'] = pixels_years_F1_MicroList
    Models_Summary['Pixels_F1_MicroList'] = pixels_F1_MicroList
    Models_Summary['Years_F1_MicroList'] = years_F1_MicroList

    Models_Summary['Pixels_Years_log_loss'] = pixels_years_logLossList
    Models_Summary['Pixels_log_loss'] = pixels_logLossList
    Models_Summary['Years_log_loss'] = years_logLossList
    
    Models_Summary['Pixels_Years_recall_binaryList'] = pixels_years_recall_binaryList
    Models_Summary['Pixels_recall_binaryList'] = pixels_recall_binaryList
    Models_Summary['Years_recall_binaryList'] = years_recall_binaryList
    
    Models_Summary['Pixels_Years_recall_MacroList'] = pixels_years_recall_MacroList
    Models_Summary['Pixels_recall_MacroList'] = pixels_recall_MacroList
    Models_Summary['Years_recall_MacroList'] = years_recall_MacroList
    
    Models_Summary['Pixels_Years_recall_MicroList'] = pixels_years_recall_MicroList
    Models_Summary['Pixels_recall_MicroList'] = pixels_recall_MicroList
    Models_Summary['Years_recall_MicroList'] = years_recall_MicroList
    
    Models_Summary['Pixels_Years_jaccard_binaryList'] = pixels_years_jaccard_binaryList
    Models_Summary['Pixels_jaccard_binaryList'] = pixels_jaccard_binaryList
    Models_Summary['Years_jaccard_binaryList'] = years_jaccard_binaryList
    
    Models_Summary['Pixels_Years_jaccard_MacroList'] = pixels_years_jaccard_MacroList
    Models_Summary['Pixels_jaccard_MacroList'] = pixels_jaccard_MacroList
    Models_Summary['Years_jaccard_MacroList'] = years_jaccard_MacroList
    
    Models_Summary['Pixels_Years_jaccard_MicroList'] = pixels_years_jaccard_MicroList
    Models_Summary['Pixels_jaccard_MicroList'] = pixels_jaccard_MicroList
    Models_Summary['Years_jaccard_MicroList'] = years_jaccard_MicroList
    
    Models_Summary['Pixels_Years_roc_auc_MacroList'] = pixels_years_roc_auc_MacroList
    Models_Summary['Pixels_roc_auc_MacroList'] = pixels_roc_auc_MacroList
    Models_Summary['Years_roc_auc_MacroList'] = years_roc_auc_MacroList
    
    Models_Summary['Pixels_Years_roc_auc_MicroList'] = pixels_years_roc_auc_MicroList
    Models_Summary['Pixels_roc_auc_MicroList'] = pixels_roc_auc_MicroList
    Models_Summary['Years_roc_auc_MicroList'] = years_roc_auc_MicroList
    
    Models_Summary['Pixels_Years_average_precision'] = pixels_years_average_precisionList
    Models_Summary['Pixels_average_precision'] = pixels_average_precisionList
    Models_Summary['Years_average_precision'] = years_average_precisionList
    

    pickling_on = open(outPath + "logGAM_2dim.pickle", "wb")
    #pickle.dump([combined_Data, target_Data, Models_Summary, Models, excluded_Years], pickling_on)
    
    pickle.dump([combined_Data, target_Data,  models, excluded_Years], pickling_on)
    pickling_on.close

    Models_Summary.to_csv(outPath + "Model_Summary_logGAM.csv")

    #return combined_Data, target_Data, Models_Summary, Models, excluded_Years

    return combined_Data, target_Data, Models_Summary, models, excluded_Years


def R_Gam_Summary_old(combined_Data, target_Data,
                        DataFields, outPath,
                        splineType = 'cs', # list for creating space for identifing optimal wifggliness penalization:
                        familyType = "binomial" #where first value indicates minimum penalty, second indicates max penalty, and 3rd value indicates number of values
                        ):
    '''Conduct logistic regressions on the data, with k-fold cross-validation conducted independently 
        across both years and pixels. 
        Returns a variety of diagnostics of model performance (including f1 scores, recall, and average precision) 
        when predicting fire risk at 
        A) locations outside of the training dataset
        B) years outside of the training dataset
        C) locations and years outside of the training dataset

      Returns a list of objects, consisting of:
        0: Combined_Data file with testing/training groups labeled
        1: Target Data file with testing/training groups labeled
        2: summary dataFrame of MSE and R2 for each model run
            (against holdout data representing either novel locations, novel years, or both)
        3: list of elastic net models for use in predicting Fires in further locations/years
        4: list of list of years not used in model training for each run
    :param combined_Data: dataFrame including all desired explanatory factors 
            across all locations & years to be used in training model
    :param target_Data: dataFrame including observed fire occurrences 
            across all locations & years to be used in training model
    :param Datafields: list of explanatory factors to be intered into model
    :param outPath: desired output location for predicted fire risk files (csv, pickle, and tif)
    :return:  returns a list of all models, accompanied by a list of years being predicted 
            - note - return output is equivalent to data exported as models.pickle
    '''
    
    mgcv = importr('mgcv') # import mgcv library from r
    stats = importr('stats') # import stats library from r
    base = importr('base') # import base library from r
    
    formula = 'value~'
    for iterparam in DataFields:
        formula += " + s(" + iterparam + ', bs = \"' + splineType + '")'
    
    
    combined_Data['value'] = target_Data['value']
    r_train = dataFrame_to_r(combined_Data)
    
    model = mgcv.gam(formula = base.eval(base.parse(text=formula)),family = base.eval(base.parse(text=familyType)), data = r_train)
    
    
    #print model summary as text file
    summary =str(base.summary(model))
    #print(type(summary))
    text_file = open(outPath + "summary.txt", "w")
    n = text_file.write(summary)
    text_file.close()
    
    
    #get plot output and convert it into graphs
    rplot = robjects.r('plot.gam')
    q = rplot(model)
    
    '''each set will be one item in the list - 
         within each set, the items will then be ordered as follows: 
    0: x values corresponding to fit & se
    1: True/False value pertaining to scale (ignore)
    2: standard error (2* se)
    3: raw (for all pixels)
    4: string of X label
    5: string of y label
    6: ignore
    7: multiplier of SE (ignore, defaults to 2)
    8: x limits (ignore?)
    9: fit (actual values to use)
    10: True/False value indicating whether it should be plotted (ignore)
'''
    
    for i in range(0,len(q)):
    
        x_Data=np.asarray(q[i][0])
        y_Data=np.asarray(q[i][9]).flatten()
        se_Data = np.asarray(q[i][2])
        upper_Data = np.add(y_Data, se_Data)
        lower_Data = np.subtract(y_Data, se_Data)
        print(x_Data.shape)
        print(y_Data.shape)
        print(se_Data.shape)
        print(upper_Data.shape)
        x_label = str(q[i][4])
        x_label= x_label[5:-2]
        y_label = str(q[i][5])
        y_label= y_label[5:-2]
        plt.clf()

        plt.plot(x_Data, y_Data, color = 'red', linewidth = 2)
        plt.plot(x_Data, upper_Data, linewidth = 2, color = 'blue', alpha = 0.5, linestyle = "--")
        plt.plot(x_Data, lower_Data, linewidth = 2, color = 'blue', alpha = 0.5, linestyle = "--")
        plt.xlabel(x_label, fontsize = '16', fontweight = 'bold')
        plt.ylabel(y_label, fontsize = '16', fontweight = 'bold')
        plt.yticks(fontsize = '12', fontweight = 'bold')
        plt.xticks(fontsize = '12', fontweight = 'bold')
        plt.tight_layout()
        plt.savefig(outPath + "Marginal_Response_"+ DataFields[i] + ".tif")


def R_Gam_Summary(combined_Data, target_Data,
                        DataFields, outPath,
                        fullDataPath = None,
                        exampleRasterPath = None,
                        marginalMask = True,
                        splineType = 'cs', 
                        familyType = "binomial" 
                        ):
    '''Conduct logistic regressions on the data, with k-fold cross-validation conducted independently 
        across both years and pixels. 
        Returns a variety of diagnostics of model performance (including f1 scores, recall, and average precision) 
        when predicting fire risk at 
        A) locations outside of the training dataset
        B) years outside of the training dataset
        C) locations and years outside of the training dataset

      Returns a list of objects, consisting of:
        0: Combined_Data file with testing/training groups labeled
        1: Target Data file with testing/training groups labeled
        2: summary dataFrame of MSE and R2 for each model run
            (against holdout data representing either novel locations, novel years, or both)
        3: list of elastic net models for use in predicting Fires in further locations/years
        4: list of list of years not used in model training for each run
    :param combined_Data: dataFrame including all desired explanatory factors 
            across all locations & years to be used in training model
    :param target_Data: dataFrame including observed fire occurrences 
            across all locations & years to be used in training model
    :param Datafields: list of explanatory factors to be intered into model
    :param outPath: desired output location for predicted fire risk files (csv, pickle, and tif)
    :param exampleRasterPath: example raster path for creating marginal maps - should be the mask used to mask out bad values if marginalMask is set to true
    :param marginalMask: determines whether marginal maps should be masked.  If so, they will be masked using file specified in exampleDataRasterPath (assuming values of 1 are valid and of 0 are to be masked out)
    :param splineType: determines type of spline for GAM
    :param familytype: determines family of GAm to be used
    :return:  returns a list of all models, accompanied by a list of years being predicted 
            - note - return output is equivalent to data exported as models.pickle
    '''
    
    mgcv = importr('mgcv') # import mgcv library from r
    stats = importr('stats') # import stats library from r
    base = importr('base') # import base library from r
    
    formula = 'value~'
    for iterparam in DataFields:
        formula += " + s(" + iterparam + ', bs = \"' + splineType + '")'
    
    
    combined_Data['value'] = target_Data['value']
    r_train = dataFrame_to_r(combined_Data)
    
    model = mgcv.gam(formula = base.eval(base.parse(text=formula)),family = base.eval(base.parse(text=familyType)), data = r_train)
    
    
    #print model summary as text file
    summary =str(base.summary(model))
    #print(type(summary))
    text_file = open(outPath + "summary.txt", "w")
    n = text_file.write(summary)
    text_file.close()
    
    
    #get plot output and convert it into graphs
    rplot = robjects.r('plot.gam')
    q = rplot(model)
    
    '''each set will be one item in the list - 
         within each set, the items will then be ordered as follows: 
    0: x values corresponding to fit & se
    1: True/False value pertaining to scale (ignore)
    2: standard error (2* se)
    3: raw (for all pixels)
    4: string of X label
    5: string of y label
    6: ignore
    7: multiplier of SE (ignore, defaults to 2)
    8: x limits (ignore?)
    9: fit (actual values to use)
    10: True/False value indicating whether it should be plotted (ignore)
'''
    
    for i in range(0,len(q)):
        #marginal response graph
        x_Data=np.asarray(q[i][0])
        y_Data=np.asarray(q[i][9]).flatten()
        se_Data = np.asarray(q[i][2])
        upper_Data = np.add(y_Data, se_Data)
        lower_Data = np.subtract(y_Data, se_Data)
        print(x_Data.shape)
        print(y_Data.shape)
        print(se_Data.shape)
        print(upper_Data.shape)
        x_label = str(q[i][4])
        x_label= x_label[5:-2]
        y_label = str(q[i][5])
        y_label= y_label[5:-2]
        plt.clf()

        plt.plot(x_Data, y_Data, color = 'red', linewidth = 2)
        plt.plot(x_Data, upper_Data, linewidth = 2, color = 'blue', alpha = 0.5, linestyle = "--")
        plt.plot(x_Data, lower_Data, linewidth = 2, color = 'blue', alpha = 0.5, linestyle = "--")
        plt.xlabel(x_label, fontsize = '16', fontweight = 'bold')
        plt.ylabel(y_label, fontsize = '16', fontweight = 'bold')
        plt.yticks(fontsize = '12', fontweight = 'bold')
        plt.xticks(fontsize = '12', fontweight = 'bold')
        plt.tight_layout()
        plt.savefig(outPath + "Marginal_Response_"+ DataFields[i] + ".tif")
    
    #Marginal Response Map
    
    fullData = pd.read_csv(fullDataPath)
    #fullData = fullData.loc[:, DataFields]
    r_full = dataFrame_to_r(fullData)
    fullTest = stats.predict(model,r_full, type = 'terms')
    fullTest = np.asarray(fullTest)

    if marginalMask == True:
      mask = image_to_array(exampleRasterPath)
      mask = mask.reshape(fullTest.shape[0], 1)
      fullTest = np.where(mask ==1, fullTest, -9999)   


    for j in range(0, len(DataFields), 1):

        arrayToRaster(fullTest[:, j], templateRasterPath = exampleRasterPath, outPath = outPath+ "Marginal_Map_"+ DataFields[j] + ".tif")
    
        
    