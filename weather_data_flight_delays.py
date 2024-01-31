
# Jack Roberge
# Weather Metrics Flight Delay Analysis
# December 2023

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report
from sklearn import tree
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier

WEATHER_DATA_CSV = '/Users/jroberge/Desktop/DS 2500/final_project_weather_data.csv'
FLIGHT_DELAY_DATA_CSV = '/Users/jroberge/Desktop/DS 2500/final_project_flight_delay_data.csv'
FLIGHT_COUNT_DATA_CSV = '/Users/jroberge/Desktop/DS 2500/final_project_flight_count_data.csv'

OLD_COLUMN_NAMES = ["ACSH","AWND","PRCP","SNOW","SNWD","TAVG","TMAX","WT01",
                    "WT02","WT03","WT04","WT05","WT06","WT07","WT08","WT09",
                    "WT10","WT11","WT13","WT14","WT15","WT16","WT17","WT18",
                    "WT19","WT21","WT22"]
NEW_COLUMN_NAMES = ['Avg Cloudiness', 'Avg Wind Speed', 'Precipitation', 
                    'Snowfall', 'Snow Depth', 'Avg Temp', 'Max Temp', 
                    'Days of Fog', 'Days of Heavy Fog', 'Days of Thunder', 
                    'Days of Sleet', 'Days of Hail', 'Days of Rime', 
                    'Days of Dust/Ash', 'Days of Smoke', 
                    'Days of Drifting Snow', 'Days of Tornado', 
                    'Days of Damaging Wind', 'Days of Mist', 'Days of Drizzle', 
                    'Days of Freezing Drizzle', 'Days of Rain', 
                    'Days of Freezing Rain', 'Days of Snow', 
                    'Days of Other Precipitation', 'Days of Ground Fog', 
                    'Days of Freezing Fog']
FEATURES_TO_AVERAGE = ['Avg Cloudiness', 'Avg Wind Speed', 'Precipitation', 
                       'Snowfall', 'Snow Depth', 'Avg Temp', 'Max Temp']
FEATURES_TO_SUM = ['Days of Fog', 'Days of Heavy Fog', 'Days of Thunder', 
                   'Days of Sleet', 'Days of Hail', 'Days of Rime', 
                   'Days of Dust/Ash', 'Days of Smoke', 
                   'Days of Drifting Snow', 'Days of Tornado', 
                   'Days of Damaging Wind', 'Days of Mist', 'Days of Drizzle', 
                   'Days of Freezing Drizzle', 'Days of Rain', 
                   'Days of Freezing Rain', 'Days of Snow', 
                   'Days of Other Precipitation', 'Days of Ground Fog', 
                   'Days of Freezing Fog']
NON_BINARY_FEATURES = ['Avg Wind Speed', 'Precipitation', 'Snowfall', 
                       'Max Temp']
BINARY_FEATURES = ['Days of Heavy Fog', 'Days of Thunder', 'Days of Sleet', 
                     'Days of Smoke']
ALL_MODEL_FEATURES = ['Avg Wind Speed', 'Precipitation', 'Snowfall', 'Max Temp', 
                    'Days of Heavy Fog', 'Days of Thunder', 'Days of Sleet', 
                    'Days of Smoke']
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 
                                   'Nov', 'Dec']
RANDOM_STATE = 0
MAX_DEPTH = 3
STARTING_K = 1
ENDING_K = 30
NUM_SPLITS = 4  


def read_data(filename, header = 0):
    ''' takes in a csv file, with an optional header command to specify which 
    row is the header and returns a dataframe with the designated row a 
    header'''
    
    df = pd.read_csv(filename, header = header)
    return df

def convert_column_to_datetime(df, column_name, date_format):
    ''' takes in a dataframe, a column name of choice and a date string format
    and turns the designated column into a datetime format, returning the 
    dataframe '''
    
    df[column_name] = pd.to_datetime(df[column_name], format = date_format)
    return df

def rename_columns(df, original_column_names, new_column_names):
    ''' takes in a dataframe, original column names and new names 
    corresponding to those original columns in order to rename them, 
    returning a dataframe with these columns renamed '''
    
    # loop through all the columns in the original columns
    for i in range(len(original_column_names)):
        
        #rename them to correspond to the same column in the renamed list
        df.rename(columns={original_column_names[i]:new_column_names[i]}, 
                  inplace = True)
    return df
    
def group_by_column_sum(df, columns_to_group_by, columns_of_interest):
    ''' group rows of a dataframe by certain columns and then sum the remaining 
    designated columns of interest, returning a dataframe with the index
    reset according to this grouping '''
    
    grouped_df = df.groupby(columns_to_group_by)[columns_of_interest].sum()
    grouped_df = grouped_df.reset_index()
    return grouped_df

def group_by_month_avg(df, date_column, columns_of_interest):
    ''' group rows of a dataframe by month and then average the columns of 
    interest, returning a dataframe with the index reset according to this
    grouping '''
    
    monthly_averages_df = df.groupby(df[date_column].dt.to_period
                                     ('M'))[columns_of_interest].mean()
    return monthly_averages_df.reset_index()

def group_by_month_sum(df, date_column, columns_of_interest):
    ''' group rows of a dataframe by month and then sum the columns of 
    interest, returning a dataframe with the index reset according to this
    grouping '''
    
    monthly_sums_df = df.groupby(df[date_column].dt.to_period
                                 ('M'))[columns_of_interest].sum()
    return monthly_sums_df.reset_index()

def convert_columns_type_to_int(df, columns):
    ''' convert the indicated columns of the entered dataframe into 
    integer types, returning this dataframe '''
    
    for column in columns:
        df[column] = df[column].astype(int)
    return df

def drop_dataframe_rows(df, column_to_check, values_to_drop):
    ''' remove all the rows of a df that contain certain values in a 
    designated column of the dataframe, returning this dataframe '''
        
    # iterate through all the values that should not be present
    for value in values_to_drop:
        
        # remove all the rows where the column's value is equal to that value
        df = df[df[column_to_check] != value]
    return df     

def create_datetime_column(df, year_column, month_column):
    ''' create a datetime column  for a df based off of designated month and 
    year columns, returning this df '''
    
    df['Date'] = pd.to_datetime(dict(year = df[year_column], 
                                     month = df[month_column], day = 1), 
                                format = '%Y%m%d').dt.to_period('M')
    return df

def combine_dataframes(df1, df2, columns_to_join_left, columns_to_join_right):
    ''' combine (left merge) two entered dataframes into one dataframe, joining 
    them on the designated column from the left and designated column from 
    the right, returning this merged dataframe '''
    combined_df = pd.merge(df1, df2, how = 'left', 
                           left_on = columns_to_join_left, 
                           right_on = columns_to_join_right)
    return combined_df

def create_target_label_groups(df, new_column_name, num_groups, 
                               initial_column_name, labels):
    ''' create a new column for a dataframe, grouping the numerical data in
    an indicated initial column into a certain number of groups, with 
    designated labels, returning this new dataframe '''
    
    df[new_column_name] = pd.qcut(df[initial_column_name], q = num_groups, 
                                  labels = labels)
    return df

def plot_metric(df, col, units, desired_num_periods, first_period, 
                 last_period):
    ''' take in a dataframe of metrics over time, plotting the metric in a
    deisgnated column monthly average over the first 50/n length period 
    compared to the last 50/n length period for n desired_num_periods '''
    
    # split the df into desired number of periods
    df['MonthNumber'] = df.index.month
    periods = np.array_split(df, desired_num_periods)
    data = []
    for period in periods:
        
        #find the monthly averages for each (50/n)-length period
        dct = {}
        for month_number, group in period.groupby('MonthNumber'):
            avg_value = group[col].mean()
            dct[month_number] = avg_value
            
        # append the monthly period averages dcg to a list of all the periods
        data.append(dct)

    # plot the averages and months for the first (50/n)-length period
    first_period_x = list(data[0].keys())
    first_period_y = list(data[0].values())
    plt.plot(first_period_x, first_period_y, marker = 'o', linestyle = '-', color = 'b', 
             label = first_period)
    
    # plot the averages and months for the last (50/n)-length period
    last_period_x = list(data[desired_num_periods - 1].keys())
    last_period_y = list(data[desired_num_periods - 1].values())
    plt.plot(last_period_x, last_period_y, marker = 'o', linestyle = '-', color = 'r', 
             label = last_period)
    
    # label the graph, axes, lines and save the figure
    plt.xticks(ticks = last_period_x, labels = MONTHS)
    plt.xlabel('Month')
    plt.ylabel(f'{col} ({units})')
    plt.title(f'Avg Daily {col} For the First and Last {round(50/desired_num_periods, 0)} Years of the Dataset')
    plt.legend()
    plt.savefig(f'plot_{col}.png', bbox_inches = 'tight')
    plt.show()

def plot_flights(df):
    ''' take in a df of monthly flight data over decades, plotting the monthly
    number of flights delayed over that period '''
    
    # find the average flights delayed per month for all 20 years
    df['MonthNumber'] = df.index.month
    dct = {}
    for month_number, group in df.groupby('MonthNumber'):
        avg_value = group['weather_delay'].mean()
        dct[month_number] = avg_value
    
    # plot the monthly averages versus the months for each month
    x_values = list(dct.keys())
    y_values = list(dct.values())
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='darkkhaki')
    
    # label the axes, graph and save the figure 
    plt.xticks(ticks=x_values, labels = MONTHS)
    plt.xlabel('Month')
    plt.ylabel('Number of Flights Delayed')
    plt.title(f'Avg Number of Delayed Flights Over 20 Years')
    plt.savefig(f'plot_flights.png', bbox_inches='tight')
    plt.show()

def linear_regression_coefficients(feature_df, labels_df, random_state):
    ''' perform a linear regression model on features and labels, 
    returning this model as an object '''
    
    # split features and labels data into testing and training data
    X_train, X_test, y_train, y_test = train_test_split(feature_df, labels_df, 
                                                        test_size = 0.25, 
                                                        random_state = random_state)
    
    # create and train the model with the features and labels training data
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def plot_numerical_predictions_versus_actual(predictions, actual, xlabel, 
                                             ylabel, title):
    ''' plot the predicted numerical values from a model against the actual
    numerical values from the model, demonstrating with a y=x line what a 
    perfect model output would be '''
    
    # find the max and min predictions and actual values to set axes and scales
    min_value = min(min(predictions), min(actual)) - 300
    max_value = max(max(predictions), max(actual)) + 300
    
    # graph the predicted vs actual values and a y=x line to compare
    plt.scatter(actual, predictions)
    plt.plot([min_value, max_value], [min_value, max_value], linestyle = '--')

    # reset and label the axes, title the graph, save the figure
    plt.xlim(min_value, max_value)
    plt.ylim(min_value, max_value)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(f"{title}.png", bbox_inches = 'tight')
    plt.show()

def linear_regression_evaluation(model, feature_df, labels_df, predictions, 
                                 actual):
    ''' calculate evaluation metrics for an inputted linear regression model,
    including mean squared error and the r squared value, printing these out '''
    
    mse = (mean_squared_error(actual, predictions))**(0.5)
    r_squared_score = model.score(feature_df, labels_df)
    print(f"This model's mean squared error is: {mse}")
    print(f"This model's r-squared score is: {r_squared_score}")

def horizontal_bar_graph(x_values, y_values, xlabel, ylabel, title):
    ''' take in x_values and y_values, as well as labels and a title, 
    to graph a horizontal bar graph from those values, saving the figure '''
    
    # create and plot the figure
    plt.figure(figsize = (10,6))
    plt.barh(y_values, x_values)
    
    #label the axes, graph and save the figure
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(f"{title}.png", bbox_inches = 'tight')
    plt.show()    
    
def decision_tree(feature_df, labels_df, random_state, max_depth):
    ''' take in a dataframe of features and a dataframe of their corresponding
    labels as well as a random_state and max_depth, returning a decision tree 
    model trained from that data '''
    
    # split features and labels data into testing and training data
    X_train, X_test, y_train, y_test = train_test_split(feature_df, labels_df, 
                                                        test_size = 0.25, 
                                                        random_state = random_state)
    
    # create the clf decision tree object, training it to the entered data
    clf = tree.DecisionTreeClassifier(max_depth = max_depth, 
                                      random_state = random_state)
    clf.fit(X_train, y_train)
    return clf

def plot_decision_tree(clf, feature_columns, label_names, title):
    ''' create a decision tree visual based off of a fitted decision tree 
    model, names of the feature columns, names of the labels, and a title of 
    the decision tree, saving the figure'''
    
    # set the size of the figure and plot the decision tree itself
    fig = plt.figure(figsize = (25,20))
    _ = tree.plot_tree(clf, feature_names = feature_columns, 
                       class_names = label_names, filled=True)
    
    # title and save the decision tree image
    plt.title(title)
    plt.savefig(f"{title}.png", bbox_inches = "tight")
    
def predict_labels(model, features_data, labels_data, random_state):
    ''' take in a trained model, features & labels data, and a random state, 
    splitting the data and returning actual test labels as well as the model's
    predicted labels for the test features '''
    
    # split features and labels data into testing and training data
    features_train, features_test, labels_train, labels_test = train_test_split(
        features_data, labels_data, test_size = 0.25, 
        random_state = random_state)    
    
    # predict test labels based off of model
    labels_predictions = model.predict(features_test)
    return labels_predictions, labels_test

def normalize_data(df, columns = None):
    ''' normalize all or certain columns in a dataframe entered, returning
    that dataframe '''
    
    # check if there are only certain columnns to normalize
    if columns:
        normalized_df = df
        
        # normalize these specific columns
        normalized_df[columns] = (normalized_df[columns] - 
                                  normalized_df[columns].min()) / (
                                      normalized_df[columns].max() - 
                                      normalized_df[columns].min())
    
    # normalize all columns in the dataframe
    else:
        normalized_df = (df - df.min()) / (df.max() - df.min())
    return normalized_df

def knn_cross_validation(features_data, labels_data, random_state, k_values, 
                         n_splits, shuffle):
    ''' performs KFold cross validation for features and labels data across
    a list of k values entered, returning an average of accuracy scores for 
    each value of k '''
    
    # create kfold object based on entered n_splits, random_state and shuffle
    scores = []
    kf = KFold(n_splits = n_splits, random_state = random_state, 
               shuffle = shuffle)
    
    # split up features and labels data into training and testing data
    features_train, features_test, labels_train, labels_test = train_test_split(
        features_data, labels_data, random_state = random_state)
    for k in k_values:
        
        # create knn model for that value of k, fit the model to the data
        knn = KNeighborsClassifier(n_neighbors = k)
        knn.fit(features_train, labels_train)
        
        # determine model accuracy score for each kfold, appending the mean of that k_value
        kfold_score = cross_val_score(estimator = knn, X = features_data, 
                                      y = labels_data, scoring = 'accuracy', 
                                      cv = kf)
        scores.append(round(np.mean(kfold_score),6))
    return scores

def choose_k(k_values, k_performance_scores):
    ''' take in a list of k_values and their respective performance scores, 
    finding the k with the maximum performance score and returning it '''
    
    # find position of k with the max performance score
    max_k_position = k_performance_scores.index(max(k_performance_scores))
    
    # find and return this k 
    max_k_value = k_values[max_k_position]
    return max_k_value

def graph_k_accuracy_scores(k_values, accuracy_scores):
    ''' graph k values against their accuracy scores '''
    
    # graph the k values and accuracy scores
    sns.lineplot(x = k_values, y = accuracy_scores)
    
    #label the axes and graph, saving the figure 
    plt.title("K Values vs Accuracy Scores")
    plt.xlabel("K Values")
    plt.ylabel("Accuracy Scores")
    plt.savefig("k_values_accuracy_scores.png", bbox_inches = "tight")
    plt.show()

def knn_predict_labels(k_value, features_data, labels_data, random_state):
    ''' take in a desired k_value and data to return a list of the model's
    labels for generated testing data as well as the testing data's actual
    labels '''
    
    # split up features and labels data into training and testing data
    features_train, features_test, labels_train, labels_test = train_test_split(
        features_data, labels_data, random_state = random_state)
    
    # create and train the knn model to the training data
    knn = KNeighborsClassifier(n_neighbors = k_value)
    knn.fit(features_train, labels_train)
    
    # predict the labels for the testing data, returning this and actual labels
    labels_prediction = knn.predict(features_test)
    return labels_test, labels_prediction
                
def create_confusion_matrix(actual_labels, predicted_labels):
    ''' create an nxn confusion matrix based off of the actual labels versus 
    the model's predicted labels, returning this matrix '''
    
    cf_matrix = confusion_matrix(actual_labels, predicted_labels)
    return cf_matrix

def create_confusion_matrix_heatmap(confusion_matrix, labels, title):
    ''' create a heatmap for an nxn confusion using the entered 
    confusion_matrix values, a list of the labels and a title for the 
    plot '''
        
    # create a suplot, arrange the tick marks
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    
    #create the heatmap from the entered confusion_matrix array
    sns.heatmap(pd.DataFrame(confusion_matrix), annot = True, cmap = "YlGnBu", 
                fmt = 'g')
    plt.tight_layout()
    
    # label the graph, axes and tick marks, saving and showing the figure
    plt.title(title)
    plt.xlabel('Predicted Label - Relative Monthly Flight Delays')
    plt.ylabel('Actual Label - Relative Monthly Flight Delays')
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.savefig(f"{title}.png", bbox_inches = "tight")
    plt.show()
    
def create_classification_report(actual_labels, predicted_labels, target_names):
    ''' print out a classification report based off of the actual labels for
    the data and the model's predicted labels for the data, using the target
    names to label the report '''
    
    print(classification_report(actual_labels, predicted_labels, 
                                target_names = target_names))
    
def output_linear_regression(features_names, features_df, labels_df, 
                             random_state, coefficient_graph_title, 
                             predicted_vs_actual_graph_title):
    ''' run a series of functions associated with linear regression models, 
    outputting numerous graphs and metrics, in order to reduce repetitive code
    when running the model for different data sets or features '''
    
    # create the linear regression model
    lr_model = linear_regression_coefficients(features_df, labels_df, random_state)
    
    # find and graph the model's features' coefficients 
    coefficients = lr_model.coef_
    horizontal_bar_graph(coefficients, features_names, 'Metric Coefficient', 
                         'Weather Metric', coefficient_graph_title)
    
    # find and graph model's predictions for training data
    label_predictions, label_actual  = predict_labels(lr_model, features_df, 
                                                      labels_df, random_state)
    plot_numerical_predictions_versus_actual(list(label_predictions), 
                                             list(label_actual), 
                                             'Actual Monthly Flight Delays', 
                                             'Predicted Monthly Flight Delays', 
                                             predicted_vs_actual_graph_title)
    
    #evaluate the model's performance using key linear regression evaluation metrics
    linear_regression_evaluation(lr_model, features_df, labels_df, 
                                 label_predictions, label_actual)
  
def output_decision_tree(feature_names, label_names, features_df, labels_df, 
                         random_state, max_depth, decision_tree_title, 
                         label_names_backwards, confusion_matrix_title, 
                         feature_importance_graph_title):
    ''' run a series of functions associated with decision tree models, 
    outputting numerous graphs and metrics, in order to reduce repetitive code
    when running the model for different data sets or features '''
    
    # create and visualize the decision tree model
    clf = decision_tree(features_df, labels_df, random_state, max_depth)
    plot_decision_tree(clf, feature_names, label_names, decision_tree_title)
    
    # calcualte and graph the model's features' importances
    clf_feature_importance = clf.feature_importances_
    horizontal_bar_graph(clf_feature_importance, feature_names, 
                         'Metric Importance', 'Weather Metric', 
                         feature_importance_graph_title)

    # find the model's for training data
    predicted_labels, actual_labels = predict_labels(clf, features_df, 
                                                     labels_df, random_state)
    
    # find and visualize the model's confusion matrix with this data
    confusion_matrix = create_confusion_matrix(actual_labels, predicted_labels)
    create_confusion_matrix_heatmap(confusion_matrix, label_names_backwards, 
                                    confusion_matrix_title)
    
    # print out a classification report for the model
    create_classification_report(actual_labels, predicted_labels, label_names)
    
def output_knn_classification(label_names, features_df, labels_df, 
                              random_state, starting_k, ending_k, num_splits, 
                              label_names_backwards, confusion_matrix_title, 
                              optimal_chosen_k):
    ''' run a series of functions associated with knn classification models, 
    outputting numerous graphs and metrics, in order to reduce repetitive code
    when running the model for different data sets or features '''
    
    # find and graph the accuracy scores of a series of k values
    scores = knn_cross_validation(features_df, labels_df, random_state, 
                                  list(range(starting_k, ending_k + 1)), 
                                  num_splits, True)
    graph_k_accuracy_scores(list(range(starting_k, ending_k + 1)), scores)
    
    # find the model's for training data
    labels_test, labels_prediction = knn_predict_labels(optimal_chosen_k, 
                                                        features_df, labels_df, 
                                                        random_state)
    
    # find and visualize the model's confusion matrix with this data
    confusion_matrix = create_confusion_matrix(labels_test, labels_prediction)
    create_confusion_matrix_heatmap(confusion_matrix, label_names_backwards, 
                                    confusion_matrix_title)
    
    # print out a classification report for the model
    create_classification_report(labels_test, labels_prediction, label_names)

def main():
    
    # read in the weather metrics, flight delay and flight count data into dfs
    weather_df = read_data(WEATHER_DATA_CSV)
    flight_delay_df = read_data(FLIGHT_DELAY_DATA_CSV)
    flight_count_df = read_data(FLIGHT_COUNT_DATA_CSV, 1)
    
    # convert the weather data's date column into a datetime type
    weather_df = convert_column_to_datetime(weather_df, 'DATE', '%Y-%m-%d')
    
    # rename the coded weather data's columns into understandable names
    weather_df = rename_columns(weather_df, OLD_COLUMN_NAMES, NEW_COLUMN_NAMES)
    
    # sum all the monthly carrier delays #s into one monthly flight delay #
    flight_delay_df = group_by_column_sum(flight_delay_df, ['year', 'month'], 
                                          ['weather_delay', 'weather_ct'])
    
    # average the quantitative daily weather metric data into monthly averages
    monthly_weather_avgs = group_by_month_avg(weather_df, 'DATE', 
                                              FEATURES_TO_AVERAGE)
    
    # sum the binary daily weather metric data into monthly counts
    monthly_weather_counts = group_by_month_sum(weather_df, 'DATE', 
                                                FEATURES_TO_SUM)
    
    # drop 'TOTAL' yearly count rows and empty rows
    flight_count_df = drop_dataframe_rows(flight_count_df, 'Month', 
                                          ['TOTAL']).dropna()
    
    # convert year and month columns into ints
    flight_count_df = convert_columns_type_to_int(flight_count_df, 
                                                  ['Year', 'Month'])
    
    # combine the flight_delay and flight_count dataframes
    flight_df = combine_dataframes(flight_delay_df, flight_count_df, 
                                   ['year', 'month'], ['Year', 'Month'])
    
    # create a datetime column for this newly combined dataframe
    flight_df = create_datetime_column(flight_df, 'year', 'month')
    
    # combine the monthly weather avg and sum dfs into one
    weather_labels = combine_dataframes(monthly_weather_avgs, 
                                        monthly_weather_counts, ['DATE'], 
                                        ['DATE'])
    
    # finally, combine the weather and flight dataframes into one dataframe
    weather_flight_df = combine_dataframes(flight_df, weather_labels, 
                                           ['Date'], ['DATE'])

    
    # create a df specifically for plotting certain quantitative weather metrics
    weather_data_to_plot = monthly_weather_avgs[['DATE', 'Snowfall', 'Max Temp', 
                                              'Precipitation']]
    weather_data_to_plot.set_index('DATE', inplace=True)
    
    # create monthly plots for first & last five-yr intervals avgs of snowfall
    plot_metric(weather_data_to_plot, 'Snowfall', 'Inches', 10, '1973-1978', 
                 '2018-2023')
    
    # create monthly plots for first & last five-yr intervals avgs of snowfall
    plot_metric(weather_data_to_plot, 'Max Temp', 'Degrees Fahrenheit', 10, '1973-1978', 
                 '2018-2023')
    
    # create monthly plots for first & last five-yr intervals avgs of snowfall
    plot_metric(weather_data_to_plot, 'Precipitation', 'Millimeters', 10, '1973-1978', 
                 '2018-2023')

    # create a df specifically for plotting flight delays over the period
    plot_flights_data = flight_df[['weather_delay', 'Date']]
    plot_flights_data.set_index('Date', inplace=True)
    
    # plot 20-yr average monthly flight delays
    plot_flights(plot_flights_data)
    
    # define labels as number of weather-related flight delays
    flight_labels = weather_flight_df['weather_delay']
    
    # define features as weather metrics for all model features
    weather_labels = weather_flight_df[ALL_MODEL_FEATURES]
    
    # output linear regression visualizations for all model features
    output_linear_regression(ALL_MODEL_FEATURES, weather_labels, 
                             flight_labels, RANDOM_STATE, 
                             'All Weather Metric Coefficients in Linear Regression Model', 
                             'All Weather Metrics Linear Regression Actual vs Predicted Monthly Flight Delays')

    # define features as weather metrics for only quantitative weather features
    weather_labels = weather_flight_df[NON_BINARY_FEATURES]
    
    # output linear regression visualizations for only quantitative features
    output_linear_regression(NON_BINARY_FEATURES, weather_labels, 
                             flight_labels, RANDOM_STATE, 
                             'Quantitative Metric Coefficients in Linear Regression Model', 
                             'Quantitative Metrics Linear Regression Actual vs Predicted Monthly Flight Delays')

    # define features as weather metrics for only binary count weather features
    weather_labels = weather_flight_df[BINARY_FEATURES]
    
    # output linear regression visualizations for only binary count features
    output_linear_regression(BINARY_FEATURES, weather_labels, flight_labels, 
                             RANDOM_STATE, 'Binary Metric Coefficients in Linear Regression Model', 
                             'Binary Metrics Linear Regression Actual vs Predicted Monthly Flight Delays')
    
    #
    weather_flight_df = create_target_label_groups(weather_flight_df, 
                                                   'weather_delay_label', 2, 
                                                   'weather_delay', 
                                                   ['Low', 'High'])
    flight_labels = weather_flight_df['weather_delay_label']
    weather_labels = weather_flight_df[ALL_MODEL_FEATURES]
    output_decision_tree(ALL_MODEL_FEATURES, ['Low', 'High'], weather_labels, 
                         flight_labels, RANDOM_STATE, MAX_DEPTH, 
                         'Decision_Tree_v1', ['High', 'Low'], 
                         'Decision_Tree_v1_Confusion_Matrix', 
                         'Decision_Tree_v1_Feature_Importance')

    weather_flight_df = create_target_label_groups(weather_flight_df, 
                                                   'weather_delay_label', 4, 
                                                   'weather_delay', 
                                                   ['Low', 'Moderate', 'High', 
                                                    'Very High'])
    weather_labels = weather_flight_df[ALL_MODEL_FEATURES]
    flight_labels = weather_flight_df['weather_delay_label']
    output_decision_tree(ALL_MODEL_FEATURES, ['Low', 'Moderate', 'High', 
                                              'Very High'], weather_labels, 
                         flight_labels, RANDOM_STATE, MAX_DEPTH, 
                         'Decision_Tree_v2', ['Very High', 'High', 'Moderate', 
                                              'Low'], 
                         'Decision_Tree_v2_Confusion_Matrix', 
                         'Decision_Tree_v2_Feature_Importance')


    weather_labels = weather_flight_df[ALL_MODEL_FEATURES]
    normalized_weather_labels = normalize_data(weather_labels)
    OPTIMAL_CHOSEN_K = 12
    output_knn_classification(['Low', 'Moderate', 'High', 'Very High'], 
                              normalized_weather_labels, flight_labels, 
                              RANDOM_STATE, STARTING_K, ENDING_K, NUM_SPLITS, 
                              ['Very High', 'High', 'Moderate', 'Low'], 
                              'KNN_Classification_Confusion_Matrix', 
                              OPTIMAL_CHOSEN_K)

if __name__ == '__main__':
    main()