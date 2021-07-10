class DatasetSpliter(object):
    def __init__(self, dataset) -> None:
        assert type(dataset) == tuple, "Dataset must be tupble"
        self.features, self.labels = dataset

    def split(self, n_split):
        datasets = []
        dps = round(len(self.features) / n_split)
        for i in range(n_split):
            features_train_part = self.features[i * dps : i * dps + dps]
            label_train_part = self.labels[i * dps : i * dps + dps]
            datasets.append((features_train_part, label_train_part))
        return datasets
