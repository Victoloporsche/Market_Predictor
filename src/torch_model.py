from processed_data import ProcessedData
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

class TorchModel:
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

    def train_model_with_torch(self):
        model = NetworkModel()
        xtrain = torch.FloatTensor(self.xtrain)
        ytrain = torch.LongTensor(self.ytrain)
        model_loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

        epochs = 100
        final_losses = []
        for i in range(epochs):
            i=i+1
            ypred = model.forward(xtrain)
            loss = model_loss(ypred, ytrain)
            final_losses.append(loss)
            if i%10==1:
                print("Epoch number {} and the loss is {}".format(i, loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        plt.plot(range(epochs), final_losses)
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        torch.save(model, '../models/trained_model_pytoch.pt')

    def model_prediction_pytorch(self):
        xtest = torch.FloatTensor(self.xtest)
        trained_model = torch.load('../models/trained_model_pytoch.pt')
        predictions = []
        with torch.no_grad():
            for i,data in enumerate(xtest):
                ytest_pred = trained_model(data)
                predictions.append(ytest_pred.argmax().item())
        predictions_test_df = pd.DataFrame(data=predictions, columns=[self.y_column_name])
        full_test_df = pd.concat([self.ts_id[self.id_column], predictions_test_df], axis=1)
        full_test_df[self.y_column_name] = full_test_df[self.y_column_name].astype(int)
        return full_test_df


class NetworkModel(nn.Module):
    def __init__(self, input_features = 130, hidden_1=25, hidden_2=25, output_features=2):
        super().__init__()
        self.fc1 = nn.Linear(input_features, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.output_layer = nn.Linear(hidden_2, output_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output_layer(x)
        return x



