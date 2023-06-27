# Kubeflow pipeline

## Build the components using your image registry

Go to each of the components and use `build_image.sh` to have own image. Update
the value of component.yaml file with new image.

## Build pipeline

Run the `pipeline.py` script. The generated pipeline will be in the `generated`
folder

## Run pipeline

1. Upload the pipeline using the UI
2. Start a run, use for the `url` parameter
   use:


## Techniques to define tasks

1. Function based
2. Docker image based
3. Reuse of existing component from kfp repository
