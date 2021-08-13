import torch


class IMDBDataset:
    def __init__(self, reviews, sentiment):
        self.reviews = reviews
        self.sentiment = sentiment

    def __len__(self):
        return self.reviews.shape[0]

    def __getitem__(self, idx):
        return {
            "reviews": torch.tensor(self.reviews[idx, :], dtype=torch.long),
            "sentiment": torch.tensor(self.sentiment[idx], dtype=torch.float),
        }
