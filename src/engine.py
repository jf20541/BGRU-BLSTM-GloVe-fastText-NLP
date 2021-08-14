import torch


class Engine:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def loss_fn(self, outputs, targets):
        """Criterion that measures the Binary Cross Entropy between the target and the output
        Args:
            outputs (float): models prediction
            targets (float): binary target values from sentiment
        Returns:
            [float]: Loss function
        """
        return torch.nn.BCELoss()(outputs, targets.view(-1, 1))

    def train_fn(self, train_loader):
        """Loop over our training set and feed the tensor inputs to the GRU model and optimize
        Args:
            train_loader (float tensors): get batches from train_loader [reviews, targets]
        Returns:
            [float]: final_targets and final_predictions
        """
        self.model.train()
        final_targets, final_predictions = [], []
        for data in train_loader:
            # get the values from cutom dataset and convert to tensors
            reviews = reviews.to(self.device, dtype=torch.long)
            targets = targets.to(self.device, dtype=torch.float)
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # forward + backward + optimize
            outputs = self.model(reviews)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            # convert to arrays, list and append to empty list
            outputs = outputs.cpu().detach().numpy().tolist()
            targets = data["sentiment"].cpu().detach().numpy().tolist()
            final_predictions.extend(outputs)
            final_targets.extend(targets)
        return final_targets, final_predictions

    def eval_fn(self, test_loader):
        """Loop over our testing set and feed the tensor inputs to the GRU model and optimize
        Args:
            train_loader (float tensors): get batches from test_loader [reviews, targets]
        Returns:
            [float]: final_targets and final_predictions
        """
        self.model.eval()
        final_targets, final_predictions = [], []
        # disables gradient calculation
        with torch.no_grad():
            for data in test_loader:
                # get the values from cutom dataset and convert to tensors
                reviews = data["reviews"].to(self.device, dtype=torch.long)
                targets = data["sentiment"].to(self.device, dtype=torch.float)
                outputs = self.model(reviews)
                # convert to arrays, list and append to empty list
                outputs = outputs.cpu().numpy().tolist()
                targets = data["sentiment"].cpu().numpy().tolist()
                final_predictions.extend(outputs)
                final_targets.extend(targets)
        return final_targets, final_predictions
