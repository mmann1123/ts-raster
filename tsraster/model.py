from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import ElasticNet
#from sklearn.preprocessing import StandardScaler as scaler
from sklearn.metrics import r2_score
from sklearn import preprocessing
import pandas as pd
from os.path import isfile

class model(object):

    def get_data(obj, scale=False):
        '''
           :param obj: path to csv or name of pandas dataframe  with yX, or list holding dataframes [y,X]
           :param scale: should data be centered and scaled True or False
        
           :return: X_train, X_test, y_train, y_test splits 
        '''
        
        # read in inputs
        print("input should be csv or pandas dataframe with yX, or [y,X]")
        if str(type(obj)) == "<class 'pandas.core.frame.DataFrame'>":
            df = obj
        elif type(obj) == list and len(obj) == 2:
            print('reading in list')
            df = pd.concat([obj[0],obj[1]],axis=1) # join Y and X 
            df.iloc[:,~df.columns.duplicated()]  # remove any repeated columns, take first
        elif isfile(obj, ): 
            df = pd.read_csv(obj)
        else:
            print("input format not dataframe, csv, or list")
            
            
        df = df.drop(['Unnamed: 0'], axis=1)  # clear out unknown columns 
        
        # check if center and scale
        if scale == True:
            min_max_scaler = preprocessing.MinMaxScaler()
            np_scaled = min_max_scaler.fit_transform(df)
            df = pd.DataFrame(np_scaled)
        
        y = df.iloc[:,0]
        X = df.iloc[:,1:]
        X_train, X_test, y_train, y_test = tts(X, y,
                                               test_size=0.33,
                                               random_state=42) 
        
        return X_train, X_test, y_train, y_test
  

    def RandomForestReg(X_train, y_train):
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

        return RF, MSE, R_Squared

    # Not working correctly
    def GradientBoosting(self):
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


    def ElasticNet(self):
        enet = ElasticNet(alpha=0.5,
                          l1_ratio=0.7)

        model = enet.fit(X_train, y_train)
        predict_test = model.predict(X=X_test)

        mse_accuracy = model.score(X_test, y_test)
        r_squared = r2_score(predict_test, y_test)
        MSE = ("MSE = {}".format(mse_accuracy))
        R_Squared = ("R-Squared = {}".format(r_squared))

        return enet, MSE, R_Squared

#    def get_data(self):
#        df = pd.read_csv(self)
#        data = df.drop('Unnamed: 0', axis=1)
#
#        sdf = scaler.fit_transform(data)
#        d = data.shape[1]
#
#        X = pd.DataFrame(sdf[:, 0])
#        y = pd.DataFrame(sdf[:, 1:d])
#        X_train, X_test, y_train, y_test = tts(X, y,
#                                               test_size=0.33,
#                                               random_state=42)
#
#        return X_train, X_test, y_train, y_test

