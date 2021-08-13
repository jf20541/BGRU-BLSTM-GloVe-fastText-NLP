import torch


class Engine:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def loss_fn(self, outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

    def train_fn(self, train_loader):
        self.model.train()
        final_targets, final_predictions = [], []
        for data in train_loader:
            reviews = reviews.to(self.device, dtype=torch.long)
            targets = targets.to(self.device, dtype=torch.float)
            self.optimizer.zero_grad()
            outputs = self.model(reviews)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            outputs = outputs.cpu().detach().numpy().tolist()
            targets = data["sentiment"].cpu().detach().numpy().tolist()
            final_predictions.extend(outputs)
            final_targets.extend(targets)
        return final_targets, final_predictions

    def eval_fn(self, test_loader):
        self.model.eval()
        final_targets, final_predictions = [], []
        with torch.no_grad():
            for data in test_loader:
                reviews = data["reviews"]
                targets = data["sentiment"]
                reviews = reviews.to(self.device, dtype=torch.long)
                targets = targets.to(self.device, dtype=torch.float)
                outputs = self.model(reviews)
                outputs = outputs.cpu().numpy().tolist()
                targets = data["sentiment"].cpu().numpy().tolist()
                final_predictions.extend(outputs)
                final_targets.extend(targets)
        return final_targets, final_predictions
