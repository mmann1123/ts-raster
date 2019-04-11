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
from tsraster.prep import set_common_index, set_df_index,set_df_mindex, image_to_series_simple, seriesToRaster
from tsraster import random
import pickle
import numpy as np


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
  print(type(randomSearch.best_params_))
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


def RandomForestReg_2dimTest(combined_Data, target_Data, varsToGroupBy, groupVars, testGroups, DataFields, outPath, 
  params = {"n_estimators": 100, #determines number of trees to build - more is better, but potentially slows processing
  'criterion' : 'mse', #criterion for measuring the quality of a split (mse or mae)
  'max_depth': 5, #maximum depth of tree - limits tree complexity, may be worth hypertuning 
  'min_samples_split': 2, #sets minimum # of samples per split (serves similar function to min-samples leaf)
  'min_samples_leaf': 10, #sets minimum # of samples per leaf - low values may lead to capturing noise, so may be worth hypertuning
  'min_weight_fraction_leaf': 0, # The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.
  'max_features': 'auto', #max number of features to be considered in a tree (auto means no hard limit)
  'max_leaf_nodes':None, #maximum number of leaf nodes - used to limit model comlexity, similarly to max_depth or min_samples_leaf
  'min_impurity_decrease': 0,
  'bootstrap' : True,
  'oob_score' : False, #internal cross-validation - ignore in preference to custom crossval
  'n_jobs' : None, #sets # processors to use, None indicates no limit
  'random_state' : None,
  'verbose' : 0,
  'warm_start': False}, 
  cv = 10):
  '''Conduct random Forest regressions on data, with k-fold cross-validation conducted independently 
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
      3: list of random forest models for use in predicting Fires in further locations/years
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
  #selectedParams = RandomSearch_Tuner(RandomForestRegressor(), combined_Data.loc[:, DataFields], target_Data['value'], params, cv)
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
          


          pixels_years_iterOutput = RandomForestReg(trainData_X, trainData_y['value'], testData_X_pixels_years, testData_y_pixels_years['value'], selectedParams)
          pixels_iterOutput = RandomForestReg(trainData_X, trainData_y['value'], testData_X_pixels, testData_y_pixels['value'], selectedParams)
          years_iterOutput = RandomForestReg(trainData_X, trainData_y['value'], testData_X_years, testData_y_years['value'], selectedParams)

          
          Models.append(pixels_years_iterOutput)
          

          pixels_years_MSEList.append(pixels_years_iterOutput[2])
          pixels_MSEList.append(pixels_iterOutput[2])
          years_MSEList.append(years_iterOutput[2])

          pixels_years_R2List.append(pixels_years_iterOutput[3])
          pixels_R2List.append(pixels_iterOutput[3])
          years_R2List.append(years_iterOutput[3])

  
  
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


def zeroMasker(row):
    ''' used as apply statement for masking in elastic_YearPredictor
    :return: 0 if cell is masked out, PredRisk otherwise
    '''
    if row['mask'] == 0:
        return 0
    else:
        return row['PredRisk']

def elastic_YearPredictor(combined_Data_Training, target_Data_Training, preMasked_Data_Path, outPath, year_List, DataFields, mask, params):
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
        
        
        elastic_iter_Model = ElasticNet('l1_ratio' = params['l1_ratio'], alpha = params['alpha'])
        
        elastic_iter_Fit = elastic_iter_Model.fit(combined_Data_iter_train, target_Data_iter_train['value'])
        
        
        
        #seriesToRaster(predict_iter, templateRasterPath, outPath + "Pred_FireRisk_" + str(iterYear) + ".tif")

        full_X = pd.read_csv(preMasked_Data_Path + "CD_" + str(iterYear) + ".csv")
        full_X = full_X.loc[:, DataFields]
        
        data = elastic_iter_Fit.predict(X=full_X)
        print(data)
        data = pd.DataFrame(data, columns = ['PredRisk'])
        index_mask = image_to_series_simple(mask)
        data['mask'] = index_mask
        data['PredRisk_Masked'] = data.apply(zeroMasker, axis =1)
        
        data.to_csv(outPath + "Pred_" + str(iterYear) + ".csv")
        
        #output predicted risk as tiff
        seriesToRaster(data['PredRisk_Masked'], mask, outPath + "Pred_" + str(iterYear) + ".tif", noData = -9999)
        
        model_List.append([elastic_iter_Fit])
        
    pickling_on = open(outPath + "models.pickle", "wb")
    pickle.dump([model_List, year_List], pickling_on)
    pickling_on.close
        
        
    return model_List, year_List

def randomForestReg_YearPredictor(combined_Data_Training, target_Data_Training, preMasked_Data_Path, outPath, year_List, DataFields, mask, params):
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
        
        
        iter_Model =RandomForestReg(params)
        
        iter_Fit = iter_Model.fit(combined_Data_iter_train, target_Data_iter_train['value'])
        
        
        
        #seriesToRaster(predict_iter, templateRasterPath, outPath + "Pred_FireRisk_" + str(iterYear) + ".tif")

        full_X = pd.read_csv(preMasked_Data_Path + "CD_" + str(iterYear) + ".csv")
        full_X = full_X.loc[:, DataFields]
        
        data = iter_Fit.predict(X=full_X)
        print(data)
        data = pd.DataFrame(data, columns = ['PredRisk'])
        index_mask = image_to_series_simple(mask)
        data['mask'] = index_mask
        data['PredRisk_Masked'] = data.apply(zeroMasker, axis =1)
        
        data.to_csv(outPath + "Pred_" + str(iterYear) + ".csv")
        
        #output predicted risk as tiff
        seriesToRaster(data['PredRisk_Masked'], mask, outPath + "Pred_" + str(iterYear) + "RFreg.tif")
        
        model_List.append([iter_Fit])
        
    pickling_on = open(outPath + "models.pickle", "wb")
    pickle.dump([model_List, year_List], pickling_on)
    pickling_on.close
        
        
    return model_List, year_List

def XGBoostModel(X_train, y_train, X_test, y_test, string_output = False, selectedParams = {"learning_rate":0.1, "n_estimators":100}):
    '''
    Conduct elastic net regression on training data and test predictive power against test data

    :param X_train: dataframe containing training data features
    :param y_train: dataframe containing training data responses
    :return: elastic net model, MSE, R-squared
    '''

    xgbr = XGBRegressor(learning_rate = selectedParams["learning_rate"], n_estimators = selectedParams["n_estimators"])

    model = xgbr.fit(X_train, y_train)
    predict_test = model.predict(data = X_test)

    MSE = model.score(X_test, y_test)
    R_Squared = r2_score(predict_test, y_test)
    
    if string_output == True:
      MSE = ("MSE = {}".format(MSE))
      R_Squared = ("R-Squared = {}".format(R_Squared))
    

    return xgbr, MSE, R_Squared, predict_test