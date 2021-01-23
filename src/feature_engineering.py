import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class FeatureEngineering:
    def __init__(self, train, test, id_column, y_column_name):
        self.train = train
        self.test = test
        self.y_column_name = y_column_name
        self.id_column = id_column
        self.data = pd.concat([self.train, self.test], ignore_index=True)
        self.scaler = MinMaxScaler()

    def scale_features(self):
        scaling_feature = [feature for feature in self.data.columns if feature not in [self.id_column,
                                                                                       self.y_column_name]]
        scaling_features_data = self.data[scaling_feature]
        self.scaler.fit(scaling_features_data)
        self.scaler.transform(scaling_features_data)
        self.data = pd.concat([self.data[[self.id_column, self.y_column_name]].reset_index(drop=True),
                    pd.DataFrame(self.scaler.transform(self.data[scaling_feature]), columns=scaling_feature)],
                    axis=1)
        return self.data


