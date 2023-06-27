# use virtual env from nb-requirements.txt

import kfp
from kfp import dsl
from kubernetes.client.models import V1EnvVar
from kfp.onprem import use_k8s_secret
from kfp.aws import use_aws_secret
from components.preprocess import preprocess
from components.load_data import get_data_from_dvc
from components.train import train
import os
from pathlib import Path

OUTPUT_DIRECTORY = 'generated'
PROJECT_ROOT = Path(__file__).absolute().parent

 
load_data_op = kfp.components.create_component_from_func(
    func=get_data_from_dvc,
    output_component_file=os.path.join(PROJECT_ROOT, OUTPUT_DIRECTORY,
                                       'load_data-component.yaml'),
    # This is optional. It saves the component spec for future use.
    base_image='python:3.9',
    packages_to_install=['dvc', 'dvc-s3'])

preprocess_op = kfp.components.create_component_from_func(
    func=preprocess,
    output_component_file=os.path.join(PROJECT_ROOT, OUTPUT_DIRECTORY,
                                       'preprocess-component.yaml'),
    # This is optional. It saves the component spec for future use.
    base_image='python:3.9',
    packages_to_install=['pandas'])

training_op = kfp.components.create_component_from_func(
    func=train,
    output_component_file=os.path.join(PROJECT_ROOT, OUTPUT_DIRECTORY,
                                       'train-component.yaml'),
    # This is optional. It saves the component spec for future use.
    base_image='python:3.9',
    packages_to_install=['pandas', 'scikit-learn', 'mlflow', 'boto3'])

# deploy_op = kfp.components.load_component_from_file(
#    os.path.join(PROJECT_ROOT, 'components', 'deploy', 'component.yaml'))
#

@dsl.pipeline(
    name="washing_machine_pipeline",
    description="WASHING MACHINE pipeline",
)
def washing_machine_pipeline(repo_url, filename):
    load_data_task = load_data_op(repo_url, filename).apply(use_aws_secret(secret_name='aws-secret', aws_access_key_id_name='AWS_ACCESS_KEY_ID', aws_secret_access_key_name='AWS_SECRET_ACCESS_KEY', aws_region='eu-south-1'))
    preprocess_task = preprocess_op(file=load_data_task.outputs['data'])

    train_task = (training_op(input_filepath=preprocess_task.outputs['output'])
                  .add_env_variable(V1EnvVar(name='MLFLOW_TRACKING_URI',
                                             value='http://mlflow-server.kubeflow.svc.cluster.local:5000'))
                  .add_env_variable(V1EnvVar(name='MLFLOW_S3_ENDPOINT_URL',
                                             value='http://minio.kubeflow.svc.cluster.local:9000'))
                  # https://kubeflow-pipelines.readthedocs.io/en/stable/source/kfp.extensions.html#kfp.onprem.use_k8s_secret
                  .apply(use_k8s_secret(secret_name='mlpipeline-minio-artifact',
                                        k8s_secret_key_to_env={
                                            'accesskey': 'AWS_ACCESS_KEY_ID',
                                            'secretkey': 'AWS_SECRET_ACCESS_KEY',
                                        })))
 #   deploy_task = deploy_op(model_uri=train_task.output)


if __name__ == '__main__':
    pipeline_output = os.path.join(PROJECT_ROOT, OUTPUT_DIRECTORY,
                                   'washing_machine-pipeline.yaml')
    kfp.compiler.Compiler().compile(washing_machine_pipeline, pipeline_output)
    print('Generated the washing machine pipeline definition')


client = kfp.Client()
client.create_run_from_pipeline_func(
    washing_machine_pipeline,
    arguments={
        "repo_url": "https://github.com/LucaBon/washingmachine-mlops.git",
        "filename": "signal_cycles_train_win_60_data.csv",
    })
