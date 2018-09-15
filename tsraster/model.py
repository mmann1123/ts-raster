from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler as scaler
from sklearn.metrics import r2_score


class model(object):

    def get_data(self):
        df = pd.read_csv(self)
        data = df.drop('Unnamed: 0', axis=1)

        sdf = scaler.fit_transform(data)
        d = data.shape[1]

        X = pd.DataFrame(sdf[:, 0])
        y = pd.DataFrame(sdf[:, 1:d])
        X_train, X_test, y_train, y_test = tts(X, y,
                                               test_size=0.33,
                                               random_state=42)

        return X_train, X_test, y_train, y_test


    def RandomForest(self):
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

        return MSE, R_Squared

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

        return MSE, R_Squared


    def ElasticNet(self):
        enet = ElasticNet(alpha=0.5,
                          l1_ratio=0.7)

        model = enet.fit(X_train, y_train)
        predict_test = model.predict(X=X_test)

        mse_accuracy = model.score(X_test, y_test)
        r_squared = r2_score(predict_test, y_test)
        MSE = ("MSE = {}".format(mse_accuracy))
        R_Squared = ("R-Squared = {}".format(r_squared))

        return MSE, R_Squared



