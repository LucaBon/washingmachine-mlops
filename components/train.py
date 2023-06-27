from kfp.components import InputPath

        

def train(input_filepath: InputPath('CSV')) -> str:
    import os
    from pathlib import Path
    import logging
    from datetime import datetime

    import pandas as pd
    import boto3
    import numpy as np
    from urllib.parse import urlparse
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    import joblib
    import mlflow
    import mlflow.sklearn
    
    def train_model(df:pd.DataFrame, test_size:float = 0.33,
                model_type:str = 'RandomForestClassifier'):
    
    
        time_col_name = 'TIMESTAMP'
        
        datetime_col_name = 'DateTime'
        target_col_name = 'target'
        pred_col_name = target_col_name +'_pred'
    
    
        logger = logging.getLogger(__name__)
        logger.info(f'Training Model {model_type}')
        date_time_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        random_state = 42
        x = df.drop([target_col_name, time_col_name,datetime_col_name], axis=1)
        y = df[target_col_name]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

        if model_type == 'RandomForestClassifier':
	        model = RandomForestClassifier(n_estimators=1000)
	        model.fit(x_train, y_train)

	        feature_importance = model.feature_importances_
	        std_feature_importance = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

	        # train_log_filepath = model_dir_path/('train_log_'+date_time_str+".txt")
	        y_test_pred = model.predict(x_test)
	        #with open(train_log_filepath,'w', encoding = 'utf-8') as f:
	        #    f.write(f'model: {model.__repr__()}\n')
	        #    f.write(f'dataset_shape:  {str(x.shape)}\n')
	        #    f.write(f'train_shape: {str(x_train.shape)}\n')
	        #    f.write(f'test_perc: {str(x_test.shape[0]/x.shape[0])}\n')
	        #    f.write(f'test_shape: {str(x_test.shape)}\n')
	        #    f.write(f'random_state: {str(random_state)}\n')
	        #    f.write(f'Accuracy on training set is : {model.score(x_train, y_train)}\n')
	        #    f.write(f'Accuracy on test set is : {model.score(x_test, y_test)}\n')
	        #    f.write(classification_report(y_test, y_test_pred))
                #
	        #for i, col in enumerate(x.columns):
	        #    f.write(f'feature {col}: importance {feature_importance[i]} with std {std_feature_importance[i]}\n')

        #TODO: the model's name must contain the window used for features creation
        #model_filepath = model_dir_path/(model_type+'_' + date_time_str + ".joblib")
        # joblib.dump(model, model_filepath)
        # logger.info(f'Model saved to {model_filepath}')
        df[pred_col_name] = model.predict(x)
        # df.to_csv(out_filepath, index=False, encoding='utf-8')
        # logger.info(f'CSV saved to {out_filepath}')
        
        mlflow.log_metric("training_accuracy", model.score(x_train, y_train))
        mlflow.log_metric("test_accuracy", model.score(x_test, y_test))
        for i, col in enumerate(x.columns):
            mlflow.log_metric("features_importances" + str(i), feature_importance[i])

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        result = mlflow.sklearn.log_model(model, "model", registered_model_name="WashingMachineModel")
        return result
    
    
    datetime_col_name = 'DateTime'
    target_col_name = 'target'
    pred_col_name = target_col_name +'_pred'

    logger = logging.getLogger(__name__)
    logger.info('Training Model')

    df = pd.read_csv(input_filepath, parse_dates=[datetime_col_name])
    
    # create bucket
    object_storage = boto3.client(
        "s3",
        endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL"),
        config=boto3.session.Config(signature_version="s3v4"),
    )
    default_bucket_name = "mlflow"

    buckets_response = object_storage.list_buckets()
    result = [
        bucket
        for bucket in buckets_response["Buckets"]
        if bucket["Name"] == default_bucket_name
    ]
    if not result:
        object_storage.create_bucket(Bucket=default_bucket_name)

    with mlflow.start_run():
    
        result = train_model(df, 0.33)
        
        return f"{mlflow.get_artifact_uri()}/{result.artifact_path}"
