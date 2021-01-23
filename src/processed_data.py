import pandas as pd
from feature_engineering import FeatureEngineering
from feature_selection import FeatureSelection

class ProcessedData:
    def __init__(self, train, test, id_column, y_column_name):
        self.train = train
        self.test = test
        self.y_column_name = y_column_name
        self.id_column = id_column
        self.data = pd.concat([self.train, self.test], ignore_index=True)
        self.feature_engineering = FeatureEngineering(self.train, self.test, self.id_column, self.y_column_name)
        self.feature_selection = FeatureSelection(self.train, self.test, self.id_column, self.y_column_name)

    def preprocess_my_data(self, num_of_features_to_select):
        self.data = self.feature_engineering.scale_features()
        self.data = self.feature_selection.perform_feature_selection(num_of_features_to_select)
        return self.data