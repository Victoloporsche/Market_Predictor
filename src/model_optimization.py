import lightgbm as lgb
from processed_data import ProcessedData
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import pickle
import numpy as np
seed = np.random.seed(22)

class ModelOptimization:
    def __init__(self, train, test, id_column, y_column_name, num_of_features_to_select=138):
        self.num_of_features_to_select = num_of_features_to_select
        self.y_column_name = y_column_name
        self.id_column = id_column
        self.number_of_train = train.shape[0]
        self.processed_data = ProcessedData(train, test, self.id_column, self.y_column_name)
        self.data = self.processed_data.preprocess_my_data(self.num_of_features_to_select)
        self.train = self.data[:self.number_of_train]
        self.test = self.data[self.number_of_train:]
        self.ytrain_df = self.train[self.y_column_name]
        self.ytrain = self.ytrain_df.values
        self.xtrain_df = self.train.drop([self.id_column, self.y_column_name], axis=1)
        self.xtrain = self.xtrain_df.values

    def model_random_optimization(self):
        lgb_estimator = lgb.LGBMClassifier(random_state=42)
        parameters = {
            'num_leaves': [60, 70, 80, 100, 120],
            'feature_fraction': [0.5, 0.7],
            'bagging_fraction': [0.7, 0.8],
            'num_trees': [50, 80, 100]
        }
        random_search = RandomizedSearchCV(estimator=lgb_estimator,
                                       param_distributions=parameters, cv=5)
        random_search_fit = random_search.fit(self.xtrain, self.ytrain)
        pickle.dump(random_search_fit, open('../models/jane_market_model.pkl', 'wb'))

    def model_grid_optimization(self):
        lgb_estimator = lgb.LGBMClassifier(random_state=42)
        parameters = {
            'num_leaves': [60, 70, 80, 100, 120],
            'feature_fraction': [0.5, 0.7],
            'bagging_fraction': [0.7, 0.8],
            'num_trees': [50, 80, 100]
        }

        grid_search = GridSearchCV(estimator=lgb_estimator,
                                       param_grid =parameters, cv=5)
        grid_search_fit = grid_search.fit(self.xtrain, self.ytrain)
        pickle.dump(grid_search_fit, open('../models/jane_market_model.pkl', 'wb'))