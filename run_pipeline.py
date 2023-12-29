# Import necessary libraries
import mlflow
from zenml.client import Client
from pipelines.training_pipeline import train_pipeline
'''First run  mlflow server --host 127.0.0.1 --port 8080 on the terminal
    to start zenml server : zenml up --blocking   '''
if __name__ == "__main__":
    # Set the MLflow tracking URI
    remote_server_uri = "http://127.0.0.1:8080" 
    mlflow.set_tracking_uri(remote_server_uri)

    # Print the experiment tracker
    print("Iske niche wala")
    print(Client().active_stack.experiment_tracker)

    # Run the pipeline and get the pipeline run object
    pipeline_run = train_pipeline(data_path=r"D:\Projects\mlops\data\NLP.csv")
