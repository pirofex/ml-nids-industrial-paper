import itertools

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn import preprocessing


class DatasetCreator():
    """Class which creates the datasets based on the gas pipeline dataset.
    """

    def __init__(self):
        self.binary = ['command response', 'binary result', 'control scheme', 'pump', 'solenoid']
        self.categorical = ['system mode']
        self.nominal = ['address', 'function', 'length', 'gain', 'crc rate']
        self.continous = ['setpoint', 'reset rate', 'deadband', 'cycle time', 'rate', 'pressure measurement', 'time']

    def normalize(self, df1):
        """
        Normalizes the values of all nominal and continous features
        to be in the range 0 - 1. Categorical features are one-hot encoded.

        Args:
            df1 ([DataFrame]): [The data which shall be normalized.]

        Returns:
            [DataFrame]: [The normalized dataframe.]
        """
        df = df1.copy()
        min_max_scaler = preprocessing.MinMaxScaler()
        min_max = []

        for c in self.continous:
            if c in df.columns:
                min_max.append(c)

        for c in self.nominal:
            if c in df.columns:
                min_max.append(c)

        df[min_max] = min_max_scaler.fit_transform(df[min_max])

        for c in self.categorical:
            if c not in df.columns:
                continue
            one_hot = pd.get_dummies(df[c])
            df = df.drop(c, axis=1)
            df = df.join(one_hot)

        return df

    def __remove_except_process_data(self, df):
        del df['address']
        del df['function']
        del df['length']
        del df['crc rate']
        del df['command response']
        del df['time']

    def __remove_except_protocol_data(self, df):
        del df['address']
        del df['command response']
        del df['setpoint']
        del df['gain']
        del df['reset rate']
        del df['deadband']
        del df['cycle time']
        del df['rate']
        del df['control scheme']
        del df['pump']
        del df['solenoid']
        del df['pressure measurement']
        del df['time']
        del df['system mode']

    def __remove_except_traffic_mining(self, df):
        del df['length']
        del df['crc rate']
        del df['function']
        del df['setpoint']
        del df['gain']
        del df['reset rate']
        del df['deadband']
        del df['cycle time']
        del df['rate']
        del df['control scheme']
        del df['pump']
        del df['solenoid']
        del df['pressure measurement']
        del df['system mode']
        del df['command response']

    def __load_dataset(self, data_path: str):
        dataset = arff.loadarff(data_path)
        df = pd.DataFrame(dataset[0])

        del df['categorized result']
        del df['specific result']

        df['binary result'] = pd.to_numeric(df['binary result'])
        df['command response'] = pd.to_numeric(df['command response'])

        self.impute_nans(df)
        return df

    def __build_data_set(self, df):
        df_copy = df.copy()
        y = df['binary result'].values
        del df['binary result']

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)

        return {
            'name': "IanArffDataset",
            'columns': list(df.columns),
            'df': df_copy,
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'X_train_norm': self.normalize(X_train),
            'X_test_norm': self.normalize(X_test)
        }

    def full_data(self, data_path: str):
        """Creates a dataset with all features.

        Args:
            data_path (str): The path to the data file.

        Returns:
            DataFrame: A dataframe containing the data.
        """
        print("\nCreating dataset [full data] ...")
        df = self.__load_dataset(data_path)

        print(df.describe())
        return self.__build_data_set(df)

    def process_data(self, data_path: str):
        """Creates a dataset with only process data features.

        Args:
            data_path (str): The path to the data file.

        Returns:
            DataFrame: A dataframe containing the data.
        """
        print("\nCreating dataset [process data] ...")

        df = self.__load_dataset(data_path)
        self.__remove_except_process_data(df)

        print("Columns: ", df.columns)
        print(df.describe())

        return self.__build_data_set(df)

    def protocol_data(self, data_path: str):
        """Creates a dataset with only protocol data features.

        Args:
            data_path (str): The path to the data file.

        Returns:
            DataFrame: A dataframe containing the data.
        """

        print("\nCreating dataset [protocol data] ...")
        df = self.__load_dataset(data_path)
        self.__remove_except_protocol_data(df)

        print("Columns: ", df.columns)
        print(df.describe())

        return self.__build_data_set(df)

    def traffic_data(self, data_path: str):
        """Creates a dataset with only process traffic features.

        Args:
            data_path (str): The path to the data file.

        Returns:
            DataFrame: A dataframe containing the data.
        """

        print("\nCreating dataset [traffic data] ...")
        df = self.__load_dataset(data_path)
        self.__remove_except_traffic_mining(df)

        print("Columns: ", df.columns)
        print(df.describe())

        return self.__build_data_set(df)

    def impute_nans(self, df):
        """Replaces NaN values in each column with the first not-NaN value before.

        Args:
            df (DataFrame): The dataframe.
        """

        nan_columns = ['setpoint', 'gain', 'reset rate', 'deadband', 'cycle time', 'rate', 'system mode',
                       'control scheme', 'pump', 'solenoid', 'pressure measurement']

        for column in nan_columns:
            if column in df:
                print("Imputing NaNs for column", column)

                values = df[column]
                values_not_nans = np.where(~np.isnan(values))[0]
                self.impute_by_keeping_last_value(values, values_not_nans)

    def impute_by_keeping_last_value(self, features, not_nans):
        if not_nans.size == 0:
            print("No not nans found!")
            return

        first_not_nan = not_nans[0]
        features[:first_not_nan] = features[first_not_nan]

        for begin, end in self.pairwise(not_nans):
            features[begin:end] = features[begin]

        '''
            handle the case if we have to keep the value
            until the end of the data set
        '''
        last = len(features)
        last_not_nan = not_nans[-1]
        if last != last_not_nan:
            features[last_not_nan + 1:] = features[last_not_nan]

    def pairwise(self, iterable):
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)


full_path = "../data/gas-pipeline-full"
traffic_path = "../data/gas-pipeline-traffic"
protocol_path = "../data/gas-pipeline-protocol"
process_path = "../data/gas-pipeline-process"


def load_full_data():
    """Helper function which loads the full dataset.

    Returns:
        DataFrame: The dataframe.
    """
    import joblib
    return joblib.load(full_path)


def load_process_data():
    """Helper function which loads the process dataset.

    Returns:
        DataFrame: The dataframe.
    """
    import joblib
    return joblib.load(process_path)


def load_protocol_data():
    """Helper function which loads the protocol dataset.

    Returns:
        DataFrame: The dataframe.
    """
    import joblib
    return joblib.load(protocol_path)


def load_traffic_data():
    """Helper function which loads the traffic dataset.

    Returns:
        DataFrame: The dataframe.
    """
    import joblib
    return joblib.load(traffic_path)


if __name__ == '__main__':
    """Creates all dataset types with different features
       and persists them on the disk.
    """

    import joblib

    path = "../data/IanArffDataset.arff"

    dataset = DatasetCreator()

    traffic_data = dataset.traffic_data(path)
    joblib.dump(traffic_data, traffic_path, compress=9)

    data = dataset.full_data(path)
    joblib.dump(data, full_path, compress=9)

    protocol_data = dataset.protocol_data(path)
    joblib.dump(protocol_data, protocol_path, compress=9)

    process_data = dataset.process_data(path)
    joblib.dump(process_data, process_path, compress=9)
