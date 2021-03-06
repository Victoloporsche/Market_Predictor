import pandas as pd
from jane_market_model import JaneMarketModel

if __name__ == '__main__':
    train = pd.read_csv('../output/cleaned_train_data.csv')
    test = pd.read_csv('../output/cleaned_test_data.csv')

    main_model = JaneMarketModel(train, test, 'ts_id', 'action')

    main_model.kfold_cross_validation()
    main_model.show_kfold_cross_validation_result()
    main_model.model_optimization_training()
    prediction_df = main_model.model_prediction()

    main_model.train_model_with_pytorch()
    predictions_df_torch_model = main_model.predict_model_with_pytorch()

    prediction_df.to_csv('../output/submission.csv', index=None)
    predictions_df_torch_model.to_csv('../output/submission_pytorch.csv', index=None)







