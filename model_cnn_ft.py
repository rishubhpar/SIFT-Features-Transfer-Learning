from tkinter.ttk import LabeledScale
import torch
import torch.nn as nn
import pandas as pd
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt 
from clf_dataset import clf_dataset 
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report 

device = 'cuda:0'

# Class for cnn fine-tuning, we will use resnet model
class cnn_ft(nn.Module):
    def __init__(self, out_classes):
        super().__init__()
        # self.model = models.resnet18(pretrained=True)
        self.model = models.vgg16(pretrained=True).to(device) 
        
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier = torch.nn.Sequential(*(list(self.model.classifier.children())[:-1])) 
        self.fc = nn.Linear(4096, out_classes) 

        # print("model: ", self.model)
        # exit()
        # in_feats = self.model.fc.in_features
        # self.model.fc = nn.Linear(in_feats, out_classes)

    def forward(self, x):
        y = self.model(x)
        y = self.fc(y)
        return y  

# This class has custom cnn implementation with 2 CNN layers and a single fc layer after that. 
class custom_cnn(nn.Module):
    def __init__(self, out_classes):
        super().__init__()
        self.model = nn.Sequential( nn.Conv2d(3, 16, 5),   # 224x224x3 -> 220x220x16
                                    nn.ReLU(),
                                    nn.MaxPool2d(2),        # 220x220x16 -> 110x110x16
        
                                    nn.Conv2d(16, 8, 5),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2),

                                    nn.Conv2d(8, 8, 5),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2),

                                    nn.Flatten(),
                                    nn.Linear(4608, out_classes),
                                    nn.Softmax(dim=1))  

        # self.model = nn.Sequential(self.layers) 

    def forward(self, x):
        y = self.model(x)
        return y


# This module has solution for question 2b and 2c - Resnet model trained for only the last layer 
def train_cnn(train_set, test_set):
    # model = cnn_ft(6).to(device) # last layer model with vgg16 
    model = custom_cnn(6).to(device) # custom model cnn

    # train hyper-parameters
    n_epochs = 40
    batch_size = 32 
    learning_rate = 0.0001
    optim = torch.optim.Adam(model.parameters(), lr = learning_rate)
    criterion = nn.CrossEntropyLoss() 

    train_dl = DataLoader(train_set, batch_size = batch_size, shuffle = True)  
    test_dl = DataLoader(test_set, batch_size = batch_size, shuffle = True)  

    print("train set size: ", len(train_set)) 
    print("test set size: ", len(test_set))     

    overall_losses_train = []
    overall_losses_test = [] 
    overall_true_labels = []
    overall_pred_labels = []

    # Checking the weights status
    # for nm, m in model.named_parameters():
    #     print(nm, m.requires_grad)

    for e in range(n_epochs):
        model.train()
        train_batch_loss = 0
        train_batch_acc = 0 
        test_batch_loss = 0
        test_batch_acc = 0
        
        # Training loop
        for id, batch in enumerate(train_dl):
            x = batch['X'].to(device)
            y = batch['Y'].to(device)

            y_ = model(x) 
        
            optim.zero_grad()
            loss = criterion(y_, y) # Computation of loss
            loss.backward()         # Backward prop of loss 
            optim.step()            # Step taken by the optimizer to compute 

            loss_val = loss.item()
            y_label = torch.argmax(y_, dim=1)

            acc_val = (y==y_label).sum() * 1.0/ y.shape[0]

            train_batch_loss += loss_val
            train_batch_acc += acc_val.item() 

            # print("[Ep:{}] [{}/{}] - train loss: {}, train acc: {}".format(e, id, len(train_dl), loss_val, acc_val.item()))
        
        # Test loop
        model.eval()
        for id, batch in enumerate(test_dl):
            x = batch['X'].to(device)
            y = batch['Y'].to(device)

            y_ = model(x)

            loss = criterion(y_, y) # Computation of loss
            loss_val = loss.item()

            y_label = torch.argmax(y_, dim=1)

            acc_val = (y==y_label).sum() * 1.0/ y.shape[0]

            if (e==n_epochs-1): # If last epoch then compute the stats 
                overall_true_labels += list(y.detach().cpu().numpy())
                overall_pred_labels += list(y_label.detach().cpu().numpy())

            test_batch_loss += loss_val
            test_batch_acc += acc_val.item()

        train_avg_loss = train_batch_loss / len(train_dl)
        train_avg_acc = train_batch_acc / len(train_dl)
        test_avg_loss = test_batch_loss / len(test_dl)
        test_avg_acc = test_batch_acc / len(test_dl)  

        overall_losses_train.append(train_avg_loss)
        overall_losses_test.append(test_avg_loss) 

        print("[Ep:{}] - train loss: {}, test loss: {}, train acc: {}, test acc: {}".format(e, round(train_avg_loss,3), round(test_avg_loss,3),
                                    round(train_avg_acc,3), round(test_avg_acc, 3))) 

    print("Classification Stats after epoch {}: ".format(n_epochs))
    print(classification_report(overall_true_labels, overall_pred_labels)) 

    # Plotting the train losses over iterations 
    plt.plot(overall_losses_train, label='train-loss')
    plt.plot(overall_losses_test, label='test-loss')
    plt.title('Training and Testing loss over training iterations')
    plt.xlabel('iterations')
    plt.ylabel('loss_val')
    plt.legend()
    plt.savefig('./figures/custom_model.png') 


def run_main():
    train_tforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_tforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ]) 

    root_path = './assignment1_data'
    train_csv_path = './data_files/train.csv'
    test_csv_path = './data_files/test.csv'

    train_dataset = clf_dataset(root_path, train_csv_path, train_tforms)
    test_dataset = clf_dataset(root_path, test_csv_path, test_tforms)    

    train_cnn(train_dataset, test_dataset)


if __name__ == "__main__":
    run_main() 