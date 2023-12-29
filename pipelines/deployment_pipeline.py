import numpy as np
import pandas as pd
import json  
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from pipelines.utils import get_data_for_test
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters

from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model

docker_settings = DockerSettings(required_integrations=[MLFLOW])


class DeploymentTriggerConfig(BaseParameters):
    """Deployment trigger config"""

    min_accuracy: float = 0.5

@step(enable_cache=False)
def dynamic_importer() -> str:
    '''Dynamically import the MLflow prediction server 
    return "zenml.integrations.mlflow.services.MLFlowDeploymentService"'''
    # Assuming get_data_for_test() is defined or imported from the appropriate module
    data = get_data_for_test()
    return data

@step
def deployment_trigger(accuracy: float, config: DeploymentTriggerConfig):
    """Implements a simple model deployment trigger that looks at
    input model accuracy and decides if it is good enough for deployment"""
    return accuracy >= config.min_accuracy

class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    """MLflow deployment getter parameters

    Attributes:
        pipeline_name: name of the pipeline that deployed the MLflow prediction server
        step_name: the name of the step that deployed the MLflow prediction server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """

    pipeline_name: str
    step_name: str
    running: bool = True

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model"
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline.

    Args:
        pipeline_name: name of the pipeline that deployed the MLflow prediction server
        step_name: the name of the step that deployed the MLflow prediction server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """
    # get the mlflow deployer stack component 
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    # fetch existing services with the same pipeline name, step name, and model name 
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,  # Fix typo here
        model_name=model_name,
        running=running,
    )
    if not existing_services:
        raise RuntimeError(
            f"No mlflow services found {pipeline_name}, "
            f"step {pipeline_step_name} and model {model_name}. "
            f"Pipeline for the '{model_name}' model is currently running."
        )
    return existing_services[0]

@step
def predictor(service: MLFlowDeploymentService, data: str) -> np.ndarray:
    """Run an inference request against a prediction service"""
    service.start(timeout=10)  # should be a NOP if already started
    json_data = json.loads(data)  # Changed variable name to avoid overwriting
    json_data.pop("columns")
    json_data.pop("index")
    columns_for_df = [
        "description"  # Include "description"
    ]
    df = pd.DataFrame(json_data["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    input_data = np.array(json_list)
    prediction = service.predict(input_data)
    return prediction 


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    data_path: str,  # Assuming data_path is a parameter
    min_accuracy: float = 0.5,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test)
    accuracy = evaluate_model(model, X_test, y_test)
    deployment_decision = deployment_trigger(accuracy)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout,
    )
@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    data = dynamic_importer()
    service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False
    )
    prediction = predictor(service=service, data=data)
    return prediction
