from kfp.components import InputPath, OutputPath


def preprocess(file_path: InputPath('CSV'),
               output_file: OutputPath('CSV')):
    import pandas as pd
    df = pd.read_csv(file_path)
    df.to_csv(output_file)
