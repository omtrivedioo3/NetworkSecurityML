import os
import sys

from sklearn import metrics

from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object,load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data,evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
import mlflow
from urllib.parse import urlparse

# import dagshub
# dagshub.init(repo_owner='omtrivedioo3', repo_name='NetworkSecurityML', mlflow=True)

# os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/krishnaik06/networksecurity.mlflow"
# os.environ["MLFLOW_TRACKING_USERNAME"]="krishnaik06"
# os.environ["MLFLOW_TRACKING_PASSWORD"]="7104284f1bb44ece21e0e2adb4e36a250ae3251f"





class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def track_mlflow(self,best_model,classificationmetric):
        # mlflow.set_registry_uri("https://dagshub.com/krishnaik06/networksecurity.mlflow")
        # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        with mlflow.start_run():
            f1_score=classificationmetric.f1_score
            precision_score=classificationmetric.precision_score
            recall_score=classificationmetric.recall_score

            

            mlflow.log_metric("f1_score",f1_score)
            mlflow.log_metric("precision",precision_score)
            mlflow.log_metric("recall_score",recall_score)
            mlflow.sklearn.log_model(best_model,"model")
            # Model registry does not work with file store
            # if tracking_url_type_store != "file":

            #     # Register the model
            #     # There are other ways to use the Model Registry, which depends on the use case,
            #     # please refer to the doc for more information:
            #     # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            #     mlflow.sklearn.log_model(best_model, "model", registered_model_name=best_model)
            # else:
            #     mlflow.sklearn.log_model(best_model, "model")


        
    def train_model(self, X_train, y_train, x_test, y_test):

        models = {
            "Random Forest": RandomForestClassifier(verbose=1),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(verbose=1),
            "Logistic Regression": LogisticRegression(verbose=1),
            "AdaBoost": AdaBoostClassifier(),
        }

        params = {
            "Decision Tree": {
                'criterion': ['gini', 'entropy', 'log_loss'],
            },
            "Random Forest": {
                'n_estimators': [8, 16, 32, 128, 256]
            },
            "Gradient Boosting": {
                'learning_rate': [.1, .01, .05, .001],
                'subsample': [0.6, 0.7, 0.75, 0.85, 0.9],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "Logistic Regression": {},
            "AdaBoost": {
                'learning_rate': [.1, .01, .001],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            }
        }

        model_report = evaluate_models(
            X_train=X_train,
            y_train=y_train,
            X_test=x_test,
            y_test=y_test,
            models=models,
            param=params
        )

        # Best Model Selection
        best_model_score = max(model_report.values())
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]

        best_model = models[best_model_name]

        # Predictions
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(x_test)

        # Metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        train_precision = precision_score(y_train, y_train_pred)
        test_precision = precision_score(y_test, y_test_pred)

        train_recall = recall_score(y_train, y_train_pred)
        test_recall = recall_score(y_test, y_test_pred)

        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)

        # Save model
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        save_object(self.model_trainer_config.trained_model_file_path, best_model)
        save_object("final_model/model.pkl", best_model)

        # Return Proper Structured Output
        return {
            "best_model": best_model_name,
            "train_metrics": {
                "accuracy": round(train_accuracy, 4),
                "precision": round(train_precision, 4),
                "recall": round(train_recall, 4),
                "f1_score": round(train_f1, 4)
            },
            "test_metrics": {
                "accuracy": round(test_accuracy, 4),
                "precision": round(test_precision, 4),
                "recall": round(test_recall, 4),
                "f1_score": round(test_f1, 4)
            }
        }


        


       
    
    
        
    def initiate_model_trainer(self):

        train_file_path = self.data_transformation_artifact.transformed_train_file_path
        test_file_path = self.data_transformation_artifact.transformed_test_file_path

        train_arr = load_numpy_array_data(train_file_path)
        test_arr = load_numpy_array_data(test_file_path)

        x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
        x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

        metrics = self.train_model(x_train, y_train, x_test, y_test)

        return metrics