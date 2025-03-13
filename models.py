import torch.nn as nn
import torch
import torchvision.models as models


class TransferLearningModel(nn.Module):
    def __init__(self, lr, device, pretrained, num_classes=2):
        super(TransferLearningModel, self).__init__()
        self.device = torch.device(device)

        self.model = models.resnet18(pretrained=pretrained)

        for param in self.model.parameters():
            param.requires_grad = False

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        self.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=lr)

    def forward(self, x):
        return self.model(x)

    def train_model(self, dataloader, epochs):
        self.to(self.device)
        loss_list = []

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0

            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                outputs = self(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            loss_list.append(total_loss)

        return loss_list

    def make_prediction(self, image_tensor):
        self.eval()

        with torch.no_grad():
            output = self(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()

        return prediction, probabilities[0][prediction].item()

