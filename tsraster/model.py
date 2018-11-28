from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import ElasticNet
from sklearn.metrics import  accuracy_score,confusion_matrix,cohen_kappa_score
#from sklearn.preprocessing import StandardScaler as scaler
from sklearn.metrics import r2_score
from sklearn import preprocessing
import pandas as pd
from os.path import isfile


def get_data(obj, test_size=0.33,scale=False,stratify=True):
    '''
       :param obj: path to csv or name of pandas dataframe  with yX, or list holding dataframes [y,X]
       :param test_size: percentage to hold out for testing (default 0.33)
       :param scale: should data be centered and scaled True or False (default)
       :param stratify: should the sample be stratified by the dependent valueTrue or False (default)

       :return: X_train, X_test, y_train, y_test splits
    '''

    # read in inputs
    print("input should be csv or pandas dataframe with yX, or [y,X]")
    if str(type(obj)) == "<class 'pandas.core.frame.DataFrame'>":
        df = obj
    elif type(obj) == list and len(obj) == 2:
        print('reading in list')
        df = pd.concat([obj[0],obj[1]],axis=1, join='inner') # join Y and X
        df.iloc[:,~df.columns.duplicated()]  # remove any repeated columns, take first
    elif isfile(obj, ):
        df = pd.read_csv(obj)
    else:
        print("input format not dataframe, csv, or list")


    df = df.drop(['Unnamed: 0'], axis=1,errors ='ignore')  # clear out unknown columns

    # check if center and scale
    if scale == True:
        min_max_scaler = preprocessing.MinMaxScaler()
        np_scaled = min_max_scaler.fit_transform(df)
        df = pd.DataFrame(np_scaled)

    y = df.iloc[:,0]
    X = df.iloc[:,1:]
    
    if stratify==True:
        X_train, X_test, y_train, y_test = tts(X, y,
                                               test_size=test_size,
                                               stratify=y,
                                               random_state=42)
    else:
        X_train, X_test, y_train, y_test = tts(X, y,
                                               test_size=test_size,
                                               random_state=42)

    return X_train, X_test, y_train, y_test


def RandomForestReg(X_train, y_train, X_test, y_test):
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
def GradientBoosting(X_train, y_train, X_test, y_test):
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
    MSE = ("MSE = {}".format(mse_accuracy))
    R_Squared = ("R-Squared = {}".format(r_squared))

    return GBoost, MSE, R_Squared


def ElasticNet(X_train, y_train):
    enet = ElasticNet(alpha=0.5,
                      l1_ratio=0.7)

    model = enet.fit(X_train, y_train)
    predict_test = model.predict(X=X_test)

    mse_accuracy = model.score(X_test, y_test)
    r_squared = r2_score(predict_test, y_test)
    MSE = ("MSE = {}".format(mse_accuracy))
    R_Squared = ("R-Squared = {}".format(r_squared))

    return enet, MSE, R_Squared


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

 