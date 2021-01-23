from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
from processed_data import ProcessedData
from sklearn.model_selection import cross_val_score
import numpy as np
seed = np.random.seed(22)

class CrossValidation:
    def __init__(self, train, test, id_column, y_column_name, num_of_features_to_select=138):
        self.num_of_features_to_select = num_of_features_to_select
        self.y_column_name = y_column_name
        self.id_column = id_column
        self.number_of_train = train.shape[0]
        self.processed_data = ProcessedData(train, test, self.id_column, y_column_name)
        self.data = self.processed_data.preprocess_my_data(self.num_of_features_to_select)
        self.train = self.data[:self.number_of_train]
        self.test = self.data[self.number_of_train:]
        self.ytrain_df = self.train[self.y_column_name]
        self.ytrain = self.ytrain_df.values
        self.xtrain_df = self.train.drop([self.id_column, self.y_column_name], axis=1)
        self.xtrain = self.xtrain_df.values
        self.clf_models = list()
        self.intiailize_clf_models()

    def get_models(self):
        return self.clf_models

    def add(self, model):
        self.clf_models.append((model))

    def intiailize_clf_models(self):
        model = RandomForestClassifier()
        self.clf_models.append((model))

        model = ExtraTreesClassifier()
        self.clf_models.append((model))

        model = MLPClassifier()
        self.clf_models.append((model))

        model = LogisticRegression()
        self.clf_models.append((model))

        model = xgb.XGBClassifier()
        self.clf_models.append((model))

        model = lgb.LGBMClassifier()
        self.clf_models.append((model))

    def kfold_cross_validation(self):
        clf_models = self.get_models()
        models = []
        self.results = {}

        for model in clf_models:
            self.current_model_name = model.__class__.__name__

            cross_validate = cross_val_score(model, self.xtrain, self.ytrain, cv=5)
            self.mean_cross_validation_score = cross_validate.mean()
            print("Kfold cross validation for", self.current_model_name)
            self.results[self.current_model_name] = self.mean_cross_validation_score
            models.append(model)
            self.save_mean_cv_result()
            print()

    def save_mean_cv_result(self):
        cv_result = pd.DataFrame({'mean_cv_model': self.mean_cross_validation_score}, index=[0])
        file_name = "../output/cv_results/{}.csv".format(self.current_model_name.lower())
        cv_result.to_csv(file_name, index=False)
        print("CV results saved to: ", file_name)

    def show_kfold_cv_results(self):
        for clf_name, mean_cv in self.results.items():
            print("{} cross validation accuracy is {:.3f}".format(clf_name, mean_cv))

