from model_prediction import ModelPrediction
from processed_data import ProcessedData
from cross_validation import CrossValidation
from model_optimization import ModelOptimization
from torch_model import TorchModel

class JaneMarketModel:
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
        self.ytrain = self.train[self.y_column_name]
        self.xtrain = self.train.drop([self.id_column, self.y_column_name], axis=1)
        self.xtest = self.test.drop([self.id_column, self.y_column_name], axis=1)
        self.cv = CrossValidation(train, test, self.id_column, self.y_column_name,
                                             num_of_features_to_select)
        self.model_predict = ModelPrediction(train, test, self.id_column, self.y_column_name,
                                  num_of_features_to_select)
        self.model_optim = ModelOptimization(train, test, self.id_column, self.y_column_name,
                                             num_of_features_to_select)
        self.torch_nn = TorchModel(train, test, self.id_column, self.y_column_name,
                                             num_of_features_to_select)

    def kfold_cross_validation(self):
        return self.cv.kfold_cross_validation()

    def show_kfold_cross_validation_result(self):
        return self.cv.show_kfold_cv_results()

    def model_optimization_training(self):
        return self.model_optim.model_random_optimization()

    def model_prediction(self):
        return self.model_predict.predict_output()

    def train_model_with_pytorch(self):
        return self.torch_nn.train_model_with_torch()

    def predict_model_with_pytorch(self):
        return self.torch_nn.model_prediction_pytorch()