import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import seaborn as sns
import glob
from sklearn.metrics import confusion_matrix



def Random_Forest():
    # merge five activity STFT extracted features
    featues_stacked_df = pd.DataFrame()
    for i, label in enumerate (["fall", "run", "walk","sitdown","standup"]):
        filepath1 = '/Users/shreyu/Desktop/proiject/model_3/my_model/Untitled Folder/'+str(label)+'_STFT_2sec_test.csv'
        data_frame = pd.read_csv(filepath1)
        featues_stacked_df = pd.concat([featues_stacked_df, data_frame], ignore_index=True)

    # stack features annoatation to make a vector of 2 seconds
    annonation_label_df = pd.DataFrame()
    count=0
    for i in range (0,len(featues_stacked_df['Activity']),int(10)):
        count+=1
        annonation_label_df = pd.concat([annonation_label_df, featues_stacked_df['Activity'][i:i+10].mode()], ignore_index=True)


    # stack features to make a vector of 2 seconds
    DWT_features_200ms=pd.DataFrame()
    count=0
    for k in range (0,featues_stacked_df.shape[0],10):
        count+=1
        DWT_features_200ms = pd.concat([DWT_features_200ms,
                                        pd.DataFrame(np.array((featues_stacked_df.drop(columns=['Activity'])[k:k+10]))
                                                    .ravel()).T], axis=0, ignore_index=True)


    #remove all labels with no activity
    featues_stacked_df = DWT_features_200ms[~DWT_features_200ms['Activity'].str.contains('NoActivity')]
    

    # Split test train set
    x_train, x_test, y_train, y_test = train_test_split(featues_stacked_df.drop('Activity', axis=1),
                                                        featues_stacked_df['Activity'], test_size=0.20)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)



    # Define the parameter grid to search
    param_grid = {
        'n_estimators': [70,100,120],
        'max_depth': [20,25],
        'min_samples_split': [2, 5, 10],
        'max_features': ['auto', 'sqrt']
    }

    # Create a Random Forest classifier
    rf_model = RandomForestClassifier(random_state=0)

    # Perform 10-fold cross-validation with parameter grid search
    grid_search = GridSearchCV(rf_model, param_grid, cv=10, scoring='accuracy')
    grid_search.fit(x_train, y_train)

    # Print the best parameters and best score from the cross-validation
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Score:", grid_search.best_score_)

    # Evaluate the best model on the test set
    best_model = grid_search.best_estimator_
    test_accuracy = best_model.score(x_test, y_test)
    print("Test Set Accuracy:", test_accuracy)

    # Make predictions on the test data
    y_pred = best_model.predict(x_test)
    from sklearn.metrics import f1_score,recall_score,precision_score,classification_report


    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')


    plt.show()

    # Print evaluation metrics
    print(f'Accuracy: {accuracy:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'Precision: {precision:.2f}')


    class_report = classification_report(y_test, y_pred)
    print("Classification Report:\n")
    print(class_report)



    cm_1 = confusion_matrix(y_test,y_pred, labels=("fall", "run", "sitdown","walk","standup"),normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_1, annot=True, cmap='Blues', fmt=".2%", xticklabels=["fall", "run", "walk","sitdown","standup"], yticklabels=["fall", "run", "walk","sitdown","standup"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Normalized Confusion Matrix')
    plt.show()






if __name__ == "__main__":
     warnings.filterwarnings("ignore")
     Random_Forest()