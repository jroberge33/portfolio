#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jack Roberge
Bank Failure KNN Classification Analysis
November 2023
"""

# import libraries and packages to be used throughout code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score

# define file paths
INSTITUTIONS_FILE = '/Users/jroberge/Desktop/DS 2500/HW 3/institutions.csv'
BANKLIST_FILE = '/Users/jroberge/Desktop/DS 2500/HW 3/banklist.csv'

# define constant column names for ease of use
INSTITUTION_FIELDS = ['CERT', 'ASSET', 'DEP', 'DEPDOM', 'NETINC', 'OFFDOM', 'ROA', 'ROAPTX', 'ROE']
INSTITUTION_FEATURES = ['ASSET', 'DEP', 'DEPDOM', 'NETINC', 'OFFDOM', 'ROA', 'ROAPTX', 'ROE']

# define key modeling constants
RANDOM_STATE = 0
NUMBER_SPLITS = 4
STARTING_K = 4
ENDING_K = 18

# create target variable words_to_flags and flags_to_words translation dicts 
FAILURE_FLAG_KEY = {0:'Did Not Fail', 1: 'Failed'}
FAILURE_FLAG_WORDS = {'Did Not Fail': 0, 'Failed': 1}

def read_data(file_name, usecols = None, encoding = None):
    ''' take in a file pathway as well as optional encoding key and columns
    to be used, returning a dataframe of the file's columns '''
    
    file_df = pd.read_csv(file_name, usecols = usecols, encoding = encoding)
    return file_df

def merge_df(df_1, df_2, df_1_join_column, df_2_join_column, merge_type):
    ''' take in two dataframes, a column in each to join on and how to 
    join the dataframes, returning the merged dataframe '''
    
    merged_df = pd.merge(df_1, df_2, left_on = df_1[df_1_join_column], 
                         right_on = df_2[df_2_join_column], how = merge_type)
    return merged_df

def normalize_data(df, columns = None):
    ''' normalize all or certain specified columns in a dataframe entered, 
    returning that dataframe with its columns min-max normalized '''
    
    # checks if there are specific columns to normalize
    if columns:
        
        # normalizes only the specified columns rather than the whole df
        normalized_df = df
        normalized_df[columns] = (normalized_df[columns] - normalized_df[
            columns].min()) / (normalized_df[columns].max() - 
                               normalized_df[columns].min())
    
    # normalizes the whole df
    else:
        normalized_df = (df - df.min()) / (df.max() - df.min())
    return normalized_df

def add_existence_flag(df, new_column, column_existence_to_check):
    ''' takes in a df, a name for a new column, and a name for an existing 
    column, checking whether or not the existing column's value is NaN - if 
    the value is not NaN, it is flagged with a 1 in the new column created '''
    
    df[new_column] = df[column_existence_to_check].notna() * 1
    return df   

def drop_null_rows(df, columns):
    ''' drops any rows with null values in the designated columns, returning
    this new df with no null values in the indicated columns '''
    
    df = df.dropna(subset = columns)
    return df

def create_knn_model(random_state, k_value, features_data, labels_data):
    ''' takes in a random_state, k_value, features data and labels data, 
    producing and returning a knn object fitted and created with these 
    parameters and data '''
    
    # split up features and labels data into training and testing data
    features_train, features_test, labels_train, labels_test = train_test_split(
        features_data, labels_data, random_state = random_state)
    
    # create the model for that value of k and train it to the training data
    knn = KNeighborsClassifier(n_neighbors = k_value)
    knn.fit(features_data, labels_data)
    return knn


def knn_cross_validation(features_data, labels_data, random_state, k_values, 
                         n_splits, shuffle):
    
    ''' take in features and labels data as well as numerous knn parameter
    information and output a dictionary of lists with scoring metrics as keys
    and a list of the mean of that scoring metric across KFolds for each
    value of k '''
    
    # creates a dct of lists for each scoring metric 
    scoring = ['accuracy', 'recall', 'precision']
    scores = {score: [] for score in scoring}
    
    # creates the KFold object with the inputted parameters
    kf = KFold(n_splits = n_splits, random_state = random_state, 
               shuffle = shuffle)
    for k in k_values:
        
        # create a knn model for each value of k 
        knn = create_knn_model(random_state, k, features_data, labels_data)
        
        # find and append the mean score across the 4 folds for each metric
        for score in scoring:            
            kfold_score = cross_val_score(estimator = knn, X = features_data, 
                                          y = labels_data, scoring = score, 
                                          cv = kf)
            scores[score].append(round(np.mean(kfold_score),6))
    return scores

def choose_k(metric_of_choice, k_values, k_performance_scores):
    ''' determine the max value of k associated with the desired performance 
    metric of choice, returning this max value of k '''
    
    # find the position of the max value of k for that metric
    max_k_position = k_performance_scores[metric_of_choice].index(max(
        k_performance_scores[metric_of_choice]))
    
    # find and return that k value 
    max_k_value = k_values[max_k_position]
    return max_k_value

def predict_labels(k_value, features_data, labels_data, random_state):
    ''' take in feature and label data, as well as a desired k_value and 
    random state, and output a knn model's predicted labels for test features
    as well as the actual labels for that test data '''
    
    # split up features and labels data into training and testing data
    features_train, features_test, labels_train, labels_test = train_test_split(
        features_data, labels_data, random_state = random_state)
    
    # create the model for that value of k and train it to the training data
    knn = KNeighborsClassifier(n_neighbors = k_value)
    knn.fit(features_train, labels_train)
    
    # predict and output the predicted labels and actual labels for test data
    labels_prediction = knn.predict(features_test)
    return labels_test, labels_prediction

def calculate_f1_scores(actual_labels, predicted_labels, classification_labels):
    ''' calculate f_1 scores for each label class based on the entered 
    predicted and actual labels, returning a dictionary with keys being
    label class flags ('0', '1') and values being f_1 scores for that class '''
    
    # calculate f_1 scores for the predicted vs actual labels
    f1_scores = f1_score(actual_labels, predicted_labels, average = None, 
                         labels = classification_labels)
    
    # create f_1 scores dct w/ label class flag keys (0, 1) and f_1 score values
    f1_scores_labeled = {}
    for i in range(len(f1_scores)):
        f1_scores_labeled[classification_labels[i]] = round(f1_scores[i], 4)
    return f1_scores_labeled        
                
def create_confusion_matrix(actual_labels, predicted_labels):
    ''' take in actual and model predicted labels to produce and return a 
    confusion matrix based off of these scores '''
    
    cf_matrix = confusion_matrix(actual_labels, predicted_labels)
    return cf_matrix
    
def predict_specific_row(normalized_df, row_identifier, identifying_column, 
                         k_value, feature_columns, labels_column, random_state):
    
    ''' take in a normalized_df with all the information necessary and find 
    a specific row with identifying information in an identified column, 
    using the features and labels data to create a knn model and predict
    the label of that specific row, returning the predicted and actual label '''
    
    # find the desired row based on identifying info in the identifying column
    row = normalized_df[normalized_df[identifying_column] == row_identifier]
    
    # sift down the entered df into just the key features and labels data
    features_data = normalized_df[feature_columns]
    labels_data = normalized_df[labels_column]
    
    # create a knn model for the entered random_state, k_value and data
    knn = create_knn_model(random_state, k_value, features_data, labels_data)
    
    # find the features and label associated just with the desired row
    row_features = row[feature_columns]
    row_label = row[labels_column]
    
    # predict the label of the row using the previously created model
    row_prediction = knn.predict(row_features)
    
    # return the actual row label and predicted row label 
    return row_label.item(), row_prediction[0]

def create_confusion_matrix_heatmap(confusion_matrix, 
                                    class_binary_to_label_dict, title):
    ''' takes in a 2x2 confusion matrix for 2 target variable classes as well 
    as a dictionary converting binary target variable class flags (0, 1) to 
    labels, producing a confusion matrix heatmap for the data '''
    
    # sets the label names of the target variable flags using the key_to_label dict
    class_names = [class_binary_to_label_dict[0], 
                   class_binary_to_label_dict[1]]
    fig, ax = plt.subplots()
    
    # creates and sets tickmarks for the label names
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    
    # creates the heatmap for the confusion_matrix inputted
    sns.heatmap(pd.DataFrame(confusion_matrix), annot = True, cmap = "YlGnBu",
                fmt = 'g')
    plt.tight_layout()
    
    # labels the axes, visualization, tickmarks, and saves & shows the figure
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    ax.xaxis.set_ticklabels([class_binary_to_label_dict[0], 
                             class_binary_to_label_dict[1]])
    ax.yaxis.set_ticklabels([class_binary_to_label_dict[0], 
                             class_binary_to_label_dict[1]])
    plt.savefig("hw3_heatmap.png", bbox_inches = "tight")
    plt.show()
    
def plot_k_performance_metrics(metric, k_performance_scores, k_values, color):
    ''' '''
    
    # plots the k_values versus performance scores with an inputted color
    fig, ax = plt.subplots(figsize = (8,4))
    line, = ax.plot(k_values, k_performance_scores[metric], color = color)
    
    # labels the visualization and axes
    plt.title(f"k values vs knn model {metric} performance")
    plt.xlabel("k values")
    plt.ylabel(f"model {metric} performance")
    
    # determines max score and its position to be used in plotting label arrow
    ymax = max(k_performance_scores[metric])
    xpos = k_performance_scores[metric].index(max(k_performance_scores[
        metric]))
    xmax = k_values[xpos]
    
    # plot a dot on the max k value
    ax.plot(xmax, ymax, 'o', color = 'black')
    
    # controls for the arrow being off the plot figure
    if xmax > 14:
        xytext_x = xmax - 4
    else:
        xytext_x = xmax + 2
        
    # plots an arrow labeling the max k value associated with the metric 
    ax.annotate(f'Max: k = {xmax}', xy=(xmax, ymax), xycoords = 'data', 
                xytext=(xytext_x, ymax), textcoords = 'data', va='top', 
                ha='left', arrowprops=dict(facecolor=color, shrink=0.1, 
                                           width = 2))
    
    # saves and shows the figure
    plt.savefig(f"hw3_{metric}_lineplot.png", bbox_inches = "tight")
    plt.show()
        
def main():
    
    # create list of k values 
    k_values = list(range(STARTING_K, ENDING_K + 1))
    
    # read in bank institutions and bank failures files into dataframes
    institutions_df = read_data(file_name = INSTITUTIONS_FILE, 
                                usecols = INSTITUTION_FIELDS)
    failures_df = read_data(file_name = BANKLIST_FILE, encoding = 'cp1252')
    
    # merge the failures and institutions dfs on bank certification number
    merged_df = merge_df(institutions_df, failures_df, 'CERT', 'Cert ', 'left')
    
    # normalize feature columns in the merged_df 
    normalized_merged_df = normalize_data(merged_df, 
                                          columns = INSTITUTION_FEATURES)
    
    # add a flag column identifying with a '1' whether or not the bank failed 
    normalized_df_with_failure_column = add_existence_flag(
        normalized_merged_df, 'failure_flag', 'Bank Name')
    
    # remove null rows in the features data 
    normalized_df_with_no_nulls = drop_null_rows(
        normalized_df_with_failure_column, INSTITUTION_FEATURES)
    
    # set the features to institution features and labels to failure flag
    df_features = normalized_df_with_no_nulls[INSTITUTION_FEATURES]
    df_labels = normalized_df_with_no_nulls['failure_flag']
    
    # find knn performance metrics for our list of k_values
    k_performance_scores = knn_cross_validation(df_features, df_labels, 
                                                RANDOM_STATE, k_values, 
                                                NUMBER_SPLITS, True)
    
    # print out the max k value associated with each of the performance metrics
    max_accuracy_k_value = choose_k('accuracy', k_values, k_performance_scores)
    print(f"The optimal value of k if we care most about mean accuracy is: {max_accuracy_k_value}")
    max_precision_k_value = choose_k('precision', k_values, k_performance_scores)
    print(f"The optimal value of k if we care most about mean precision is: {max_precision_k_value}")
    max_recall_k_value = choose_k('recall', k_values, k_performance_scores)
    print(f"The optimal value of k if we care most about mean recall is: {max_recall_k_value}")
    
    # print out the minimum accuracy score 
    min_accuracy_value = min(k_performance_scores['accuracy'])
    print(f"The lowest mean accuracy for any value of k is: {min_accuracy_value}")
    
    # determine the optimal k to use based off of the accuracy metric
    k_accuracy = choose_k('accuracy', k_values, k_performance_scores)
    
    # find actual labels and the model's predicted labels with this value of k
    actual_labels, predicted_labels = predict_labels(k_accuracy, df_features, 
                                                     df_labels, RANDOM_STATE)

    # calculate the f_1 scores of the model's predictions using label classes
    f1_scores = calculate_f1_scores(actual_labels, predicted_labels, list(
        FAILURE_FLAG_KEY.keys()))
    
    # print out f_1 score for failed banks using the label words_to_flag dict
    print(f"The f1 score for banks that have failed is: {f1_scores[FAILURE_FLAG_WORDS['Failed']]}")
    
    # create confusion matrix for the actual, predicted labels from the accuracy model
    confusion_matrix = create_confusion_matrix(actual_labels, predicted_labels)
    print(f"The number of banks that my model predicted would not fail and indeed did not fail is: {confusion_matrix[FAILURE_FLAG_WORDS['Did Not Fail']][FAILURE_FLAG_WORDS['Did Not Fail']]}")    

    # find and report predicted and actual label for the Southern Community Bank
    row_result, row_prediction = predict_specific_row(
        normalized_df_with_no_nulls, 'Southern Community Bank', 'Bank Name', 
        k_accuracy, INSTITUTION_FEATURES, 'failure_flag', RANDOM_STATE)
    print(f"The model predicts that the Southern Community Bank {FAILURE_FLAG_KEY[row_prediction]} when in reality, the bank {FAILURE_FLAG_KEY[row_result]}")

    # find actual and predicted labels based on the max recall model 
    k_recall = choose_k('recall', k_values, k_performance_scores)
    actual_labels, predicted_labels = predict_labels(k_recall, df_features, 
                                                     df_labels, RANDOM_STATE)
    
    # create and visualize confusion matrix for the labels from the recall model
    confusion_matrix = create_confusion_matrix(actual_labels, predicted_labels)
    create_confusion_matrix_heatmap(confusion_matrix, FAILURE_FLAG_KEY, 'Confusion Matrix Max Recall knn Heatmap')
    
    # plot k_values vs performance metrics for accuracy, recall and precision
    plot_k_performance_metrics('accuracy', k_performance_scores, k_values, 
                               'green')
    plot_k_performance_metrics('recall', k_performance_scores, k_values, 
                               'red')
    plot_k_performance_metrics('precision', k_performance_scores, k_values, 
                               'blue')

if __name__ == '__main__':
    main()
