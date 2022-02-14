Instructions:

1. To run the sift module call "python3 sift_feats.py"
2. To run the knn module call "python3 knn.py"
3. To run the VGG16 classifier and the custom model run "python3 model_cnn_ft.py" 
   and select the appropriate model by uncommenting it 



Below are the detailed classification results for all the three models for reference. 


Statistics:

KNN: 
                       precision    recall  f1-score   support

            Albatross       0.95      1.00      0.98        20
           frangipani       1.00      1.00      1.00        20
             Marigold       0.95      1.00      0.98        20
            anthuriam       1.00      1.00      1.00        20
Red_headed_Woodpecker       0.95      0.90      0.92        20
   American_Goldfinch       1.00      0.95      0.97        20

             accuracy                           0.97       120
            macro avg       0.98      0.98      0.97       120
         weighted avg       0.98      0.97      0.97       120



VGG:

              precision    recall  f1-score   support

           0       1.00      1.00      1.00        20
           1       1.00      1.00      1.00        20
           2       1.00      1.00      1.00        20
           3       1.00      1.00      1.00        20
           4       1.00      1.00      1.00        20
           5       1.00      1.00      1.00        20

    accuracy                           1.00       120
   macro avg       1.00      1.00      1.00       120
weighted avg       1.00      1.00      1.00       120 


Custom model: 


              precision    recall  f1-score   support

           0       0.81      0.65      0.72        20
           1       0.78      0.90      0.84        20
           2       0.95      1.00      0.98        20
           3       1.00      1.00      1.00        20
           4       0.57      0.65      0.60        20
           5       0.71      0.60      0.65        20

    accuracy                           0.80       120
   macro avg       0.80      0.80      0.80       120
weighted avg       0.80      0.80      0.80       120