import pandas as pd

def count_missing_values(dataset_path: pd.DataFrame, output: bool = False) -> pd.DataFrame:
    """
    Count the number of missing values in each column of the dataset.

    Returns:
    pd.Series: Series containing the count of missing values for each column.
    """

    # Count missing values in each column
    missing_counts = dataset_path.isnull().sum()
    missing_counts = pd.DataFrame(missing_counts[missing_counts > 0], columns=['Missing Count'])

    if output:
        print(missing_counts)
    
    return missing_counts

if __name__ == "__main__":

    datasets = ['Arrhythmia Data Set', 'gene expression cancer RNA-Seq Data Set']

    for dataset in datasets:

        print(f'\n ----------    {dataset}     --------------\n')

        train_data_path = f"./dataset/{dataset}/train_data.csv"
        train_label_path = f"./dataset/{dataset}/train_label.csv"

        x_train = pd.read_csv(train_data_path, header = None if dataset == 'Arrhythmia Data Set' else 0)

        print(count_missing_values(x_train))

