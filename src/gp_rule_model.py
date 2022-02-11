class GasPipelineRuleModel:
    """Custom rules which predict based on certain feature values
       if a data packet is an attack or not.
    """

    def __init__(self):
        pass

    def predict(self, data):
        """Predicts a collection of network packets.

        Args:
            data (DataFrame): The data which shall be classified.

        Returns:
            list: A list of predictions, 1 means 'attack' while 0 means 'normal'.
        """
        predictions = []

        for index, row in data.iterrows():
            if (row['crc rate'] > 14000 and row['command response'] == 1) or row['crc rate'] > 16000 or row[
                'function'] > 16 or row['length'] < 16 or row['address'] != 4:
                predictions.append(1)
            else:
                predictions.append(0)

        return predictions

    def fit(self, X, y):
        pass


class ModelWrapper:
    """Wraps a model which implements the scikit-learn API.
    """

    def __init__(self, model):
        self.model = model

    def predict(self, data):
        result = self.model.predict(data)

        results = []

        for r in result:
            if r >= 0.5:
                results.append(1)
            else:
                result.append(0)

        return results
