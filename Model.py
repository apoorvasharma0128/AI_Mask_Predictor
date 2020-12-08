import torch
import torch.nn as nn
import torch.nn.functional as F
import DataManager as dm
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np

import torchvision
import torchvision.transforms as transforms

num_epochs = 5
num_classes = 3
learning_rate = 0.001
transformMsg=dm.transformImages()
training =os.getcwd()+"/Database/training-data"

dataset=dm.loadImages1123(training)
classes = sorted(os.listdir(training))

def model_create():

    class Net(nn.Module):
        def __init__(self, num_classes=3):
            super(Net, self).__init__()
            self.conv1 = self.hidden_layers(3,16)
            self.conv2 = self.hidden_layers(16,32)
            self.conv3 = self.hidden_layers(32,64)
            self.conv4 = self.hidden_layers(64,128)
            self.conv5 = nn.Sequential(
                torch.nn.Linear(128*8*8, 3),
                torch.nn.Sigmoid()
            )


        def hidden_layers(self, in_channel, out_channel):
            return nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, 1, 1),
                nn.MaxPool2d(2),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1))

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = x.view(-1, 128 * 8 * 8)
            x = self.conv5(x)
            return x

    model = Net(num_classes=len(classes))
    print("Model Summary")
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

def test_model(model, test_loader,fold):
    model.eval()
    actual = []
    predictions = []
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            for label in labels.data.numpy():
                actual.append(label)
            for prediction in outputs.data.numpy().argmax(1):
                predictions.append(prediction)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model: {} %'
              .format((correct / total) * 100))


        perform_evaluation(actual,predictions,fold)


def plot_ConfusionMatrix(confusion_matrix,label):

    figure = plt.figure()
    plt.title(label)
    ax = figure.add_subplot(111)

    ax.matshow(confusion_matrix, interpolation ='nearest')
    for (x, y), z in np.ndenumerate(confusion_matrix):
        ax.text(x, y, '{:0.1f}'.format(z), ha='center', va='center')
    ax.set_xticklabels(['']+classes)
    ax.set_yticklabels(['']+classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def perform_evaluation(actual,predictions,fold):
    print("Evaluating fold",num_fold)
    cm = np.array(confusion_matrix(actual, predictions))
    cr =classification_report(actual,predictions)
    print("Confusion Matrix")
    print(cm)
    print("Classification Report")
    print(cr)
    label="Confusion Matrix for fold" + str(fold)
    plot_ConfusionMatrix(cm,label)

model=model_create()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_criteria = nn.CrossEntropyLoss()
from sklearn.model_selection import KFold
kfold = KFold(10, True, None)
num_fold=1
for train_idx, test_idx in kfold.split(dataset):
    print("Running Fold",num_fold)
    print('Training Model')
    train_dataset = torch.utils.data.Subset(dataset,train_idx)
    val_set = torch.utils.data.Subset(dataset,test_idx)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=50,num_workers=0,shuffle=True)
    val_set_loader = torch.utils.data.DataLoader(val_set,batch_size=50,num_workers=0,shuffle=True)
    model_train(model, train_loader, optimizer, num_epochs)
    print('Testing Model')
    test_model(model, val_set_loader,num_fold)
    num_fold=num_fold+1
