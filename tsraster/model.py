from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import GroupShuffleSplit

from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import  accuracy_score,confusion_matrix,cohen_kappa_score
#from sklearn.preprocessing import StandardScaler as scaler
from sklearn.metrics import r2_score
from sklearn import preprocessing
import pandas as pd
from os.path import isfile
from tsraster.prep import set_common_index, set_df_index,set_df_mindex



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



def RandomForestReg(X_train, y_train, X_test, y_test):
    '''
    Conduct random forest regression on training data and test predictive power against test data

    :param X_train: dataframe containing training data features
    :param y_train: dataframe containing training data responses
    :param X_test: dataframe containing test data features
    :param X_train: dataframe containing test data features
    :return: Random Forest Model, dataframe of predicted responses for test dataset, mse of model on test data, r2 of model on test data

    '''

    RF = RandomForestRegressor(n_estimators=100,
                               criterion="mse",
                               max_depth=10,
                               min_samples_leaf=5,
                               min_samples_split=5,
                               random_state=42)

    model = RF.fit(X_train, y_train)
    predict_test = model.predict(X=X_test)

    mse_accuracy = model.score(X_test, y_test)
    r_squared = r2_score(predict_test, y_test)
    MSE = ("MSE = {}".format(mse_accuracy))
    R_Squared = ("R-Squared = {}".format(r_squared))

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
def GradientBoosting(X_train, y_train, X_test, y_test, string_output = False):
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
    else if string_output == False:
      MSE = mse_accuracy
      R_Squared = r_squared

    return GBoost, MSE, R_Squared


def ElasticNetModel(X_train, y_train, X_test, y_test):
    '''
    Conduct elastic net regression on training data and test predictive power against test data

    :param X_train: dataframe containing training data features
    :param y_train: dataframe containing training data responses
    :return: elastic net model, MSE, R-squared
    '''

    enet = ElasticNet(alpha=0.5,
                      l1_ratio=0.7)

    model = enet.fit(X_train, y_train)
    predict_test = model.predict(X=X_test)

    mse_accuracy = model.score(X_test, y_test)
    r_squared = r2_score(predict_test, y_test)
    MSE = ("MSE = {}".format(mse_accuracy))
    R_Squared = ("R-Squared = {}".format(r_squared))

    return enet, MSE, R_Squared

def ElasticNetCVModel(X_train, y_train, X_test, y_test):
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
    MSE = ("MSE = {}".format(mse_accuracy))
    R_Squared = ("R-Squared = {}".format(r_squared))

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

 