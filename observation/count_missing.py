import pandas as pd

def count_missing_values(dataset_path: pd.DataFrame, output: bool = False) -> pd.DataFrame:
    """
    Count the number of missing values in each column of the dataset.

    Returns:
    pd.DataFrame: DataFrame containing the count of missing values for each column.
    """

    # Count missing values in each column
    missing_counts = dataset_path.isnull().sum()
    missing_counts = pd.DataFrame(missing_counts[missing_counts > 0], columns=['Missing Count'])

    if output:
        if missing_counts.empty:
            print("No missing values found in the dataset.")
        else:
            for index in missing_counts.index:
                print(f"Column {index} has {missing_counts.loc[index, 'Missing Count']: 4d} missing values.")
                print("With unique values:")
                print(dataset_path[index].unique())
    
    return missing_counts

if __name__ == "__main__":

    datasets = ['Arrhythmia Data Set', 'gene expression cancer RNA-Seq Data Set']

    for dataset in datasets:

        print(f'\n ----------    {dataset}     --------------\n')

        train_data_path = f"./dataset/{dataset}/train_data.csv"
        train_label_path = f"./dataset/{dataset}/train_label.csv"

        x_train = pd.read_csv(train_data_path, header = None if dataset == 'Arrhythmia Data Set' else 0)

        count_missing_values(x_train, True)

