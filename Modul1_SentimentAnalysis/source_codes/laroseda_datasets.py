from torch.utils.data import Dataset


class LarosedaDatasetTrain(Dataset):
    def __init__(self, train, labels):
        super().__init__()
        self.vectorized_data, self.labels = train, labels
    def __len__(self):
        return len(self.vectorized_data)

    def __getitem__(self, idx):
        tokenized_text_input_to_model = self.vectorized_data[idx]
        label = self.labels[idx]
        return tokenized_text_input_to_model, label

class LarosedaDatasetTest(Dataset):
    def __init__(self, train):
        super().__init__()
        self.vectorized_data = train
    def __len__(self):
        return len(self.vectorized_data)

    def __getitem__(self, idx):
        tokenized_text_input_to_model = self.vectorized_data[idx]
        return tokenized_text_input_to_model