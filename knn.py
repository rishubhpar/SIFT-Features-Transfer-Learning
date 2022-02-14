from tkinter.ttk import LabeledScale
import torch
import torch.nn as nn
import pandas as pd
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from clf_dataset import clf_dataset
import numpy as np
from sklearn.metrics import classification_report

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# This module will extract feature for a given input x and the vgg backbone models
def extract_features(model, model_fc, x):
    x = x.to(device)
    cnn_feats = model.features(x)
    # print("cnn feats shape:" ,cnn_feats.shape)
    avg_pool = model.avgpool(cnn_feats).view(-1)
    # print("avg pool shape:", avg_pool.shape)
    fc_feats = model_fc(avg_pool)
    # print("fc feats shape: ", fc_feats.shape)
    
    return fc_feats  

# This function will find the closest match to a given new test feature and the set of training features 
def find_nn(test_feat, train_features):
    min_dst = 10000
    min_id = -1

    for id in range(0, len(train_features)):
        train_feat = train_features[id]
        dst = torch.nn.MSELoss()(test_feat, train_feat)
        if (dst < min_dst):
            min_dst = dst
            min_id = id

    return min_id

# This function has the code solution for the question 2a) part - Implementation of KNN model 
def knn(train_set, test_set):
    # Backbone model
    model = models.vgg16(pretrained=True).to(device) 
    # Classifier head of the backbone model 
    model_fc = torch.nn.Sequential(*(list(model.classifier.children())[:-3]))
    for param in model.parameters():
        param.requires_grad = False
    for param in model_fc.parameters():
        param.requires_grad = False



    train_dl = DataLoader(train_set, batch_size = 1, shuffle = False) 
    test_dl = DataLoader(test_set, batch_size = 1, shuffle = False) 

    train_features = [] 
    test_features = []

    for id, data in enumerate(train_dl):
        train_feat = extract_features(model, model_fc, data['X']).cpu()
        train_features.append(train_feat)

    print("Features extracted for training set. ", len(train_features))
    for id, data in enumerate(test_dl):
        test_feat = extract_features(model, model_fc, data['X']).cpu()
        test_features.append(test_feat)
        
    print("Features extracted for the test set. ", len(test_features))

    test_preds = []
    test_true = []
    for id in range(0, len(test_features)):
        test_feat = test_features[id]
        # Id for the minimum distance data point in the training set
        min_id = find_nn(test_feat, train_features)
        pred_test_label = train_set[min_id]['Y']  
        true_test_label = test_set[id]['Y']
        
        test_preds.append(pred_test_label)
        test_true.append(true_test_label)

    test_preds = np.array(test_preds)
    true_labels = np.array(test_true)
    # print("test preds: ", test_preds)
    # print()
    # print("true preds: ", true_labels) 

    targets = ['Albatross', 'frangipani', 'Marigold', 'anthuriam', 'Red_headed_Woodpecker', 'American_Goldfinch'] 
    print("Classification report KNN: ") 
    print(classification_report(true_labels, test_preds, target_names=targets))
    


# Main function to run the model 
def run_main():
    root_path = './assignment1_data'
    train_csv_path = './data_files/train.csv'
    test_csv_path = './data_files/test.csv'

    tforms = transforms.Compose( [ transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_dataset = clf_dataset(root_path, train_csv_path, tforms)
    test_dataset = clf_dataset(root_path, test_csv_path, tforms) 

    knn(train_dataset, test_dataset)


if __name__ == "__main__":
    run_main()