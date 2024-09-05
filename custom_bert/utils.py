import pandas as pd
def extract_and_save_columns(input_csv_path, output_csv_path):
    """
    Given the made csv files make csv files for the bert training.
    """
    df = pd.read_csv(input_csv_path)
    selected_columns = df[['transcription_text', 'class_label', 'converted_mmse']]
    selected_columns.to_csv(output_csv_path, index=False)
    print(f"Data saved to {output_csv_path}")