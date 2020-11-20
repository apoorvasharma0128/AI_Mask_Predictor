import torch
import torch.nn as nn
import torch.nn.functional as F
import DataManager as dm
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np

num_epochs = 4
num_classes = 3
learning_rate = 0.001
dm.transformImages()
training =os.getcwd()+"/Database/training-data"
test_loader,train_loader=dm.loadImages(training)
classes = sorted(os.listdir(training))

def model_create():

    class Net(nn.Module):


        # Defining the Constructor
        def __init__(self, num_classes=3):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2)
            self.drop = nn.Dropout2d(p=0.2)
            self.fc = nn.Linear(in_features=32 * 32 * 24, out_features=num_classes)

        def forward(self, x):
            x = F.relu(self.pool(self.conv1(x)))
            x = F.relu(self.pool(self.conv2(x)))
            x = F.dropout(self.drop(x), training=self.training)
            x = x.view(-1, 32 * 32 * 24)
            x = self.fc(x)
            return torch.log_softmax(x, dim=1)

    # Create an instance of the model class and allocate it to the device
    model = Net(num_classes=len(classes))
    print(model)
    return model

def model_train(model, train_loader, optimizer, num_epochs):
    # Set the model to training mode
    model.train()
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
            loss = loss_criteria(outputs, labels)
            loss_list.append(loss.item())
            # Backprop and optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Train accuracy
            total = labels.size(0)
            _,predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)


        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.
              format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                     (correct / total) * 100))

def test_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model on the 10000 test images: {} %'
              .format((correct / total) * 100))

model=model_create()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_criteria = nn.CrossEntropyLoss()
print('Training Model')
model_train(model, train_loader, optimizer, num_epochs)
for epoch in range(num_epochs):
    test_model(model, test_loader)


actual = []
predictions = []
model.eval()
for data, target in test_loader:
    for label in target.data.numpy():
        actual.append(label)
    for prediction in model(data).data.numpy().argmax(1):
        predictions.append(prediction)

cm = np.array(confusion_matrix(actual, predictions))
cr =classification_report(actual,predictions)
print("Confusion Matrix")
print(cm)
print("Classification Report")
print(cr)

def plot_ConfusionMatrix(confusion_matrix):
    figure = plt.figure()
    axes = figure.add_subplot(111)
    axes.matshow(cm, interpolation ='nearest')
    for (i, j), z in np.ndenumerate(cm):
        axes.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
    axes.set_xticklabels(['']+classes)
    axes.set_yticklabels(['']+classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
plot_ConfusionMatrix(cm)

