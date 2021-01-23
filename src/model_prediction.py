from model_optimization import ModelOptimization
from processed_data import ProcessedData
import pandas as pd
import pickle

class ModelPrediction:
    def __init__(self, train, test, id_column, y_column_name, num_of_features_to_select=138):
        self.num_of_features_to_select = num_of_features_to_select
        self.y_column_name = y_column_name
        self.id_column = id_column
        self.number_of_train = train.shape[0]
        self.processed_data = ProcessedData(train, test, self.id_column, self.y_column_name)
        self.data = self.processed_data.preprocess_my_data(self.num_of_features_to_select)
        self.train = self.data[:self.number_of_train]
        self.test = self.data[self.number_of_train:]
        self.ts_id = self.test[self.id_column].reset_index()
        self.ytrain_df = self.train[self.y_column_name]
        self.ytrain = self.ytrain_df.values
        self.xtrain_df = self.train.drop([self.id_column, self.y_column_name], axis=1)
        self.xtrain = self.xtrain_df.values
        self.xtest_df = self.test.drop([self.id_column, self.y_column_name], axis=1)
        self.xtest = self.xtest_df.values
        self.model_optim = ModelOptimization(train, test, self.id_column, self.y_column_name,
                                             num_of_features_to_select)

    def predict_output(self):
        trained_and_optimized_model = pickle.load(open('../models/jane_market_model.pkl', 'rb'))
        y_predict = trained_and_optimized_model.predict(self.xtest)
        predictions_test_df = pd.DataFrame(data=y_predict, columns=[self.y_column_name])
        full_test_df  = pd.concat([self.ts_id[self.id_column], predictions_test_df], axis=1)
        full_test_df[self.y_column_name] = full_test_df[self.y_column_name].astype(int)
        return full_test_df