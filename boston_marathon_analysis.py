#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jack Roberge
Boston Marathon Analysis
10/20/2023
"""

import csv
import os
import matplotlib.pyplot as plt
import statistics
from scipy import stats
import seaborn as sns

FIRST_YEAR = 2010
LAST_YEAR = 2023
YEARS_EXCLUDED = [2020]
FILE_DIR = "marathon_data"
FILE_NAME = "boston_marathon_"
TIME_HOURS_HEADER = "OfficialTime"
TIME_SECONDS_HEADER = "OfficialTimeSeconds"
AGE_HEADER = "AgeOnRaceDay"
COUNTRY_RES_ABBR_HEADER = "CountryOfResAbbrev"
GENDER_HEADER = "Gender"


def read_csv(file_path, file_name):
    ''' given the name of a csv file, return its contents as a list of lists,
        including the header.'''
        
    #use inputted file directory and name to find file path 
    data = []
    filename = file_path + "/" + file_name
    
    #open and read file into  a list of lists that is returned
    with open(filename, "r") as infile:
        csvfile = csv.reader(infile)
        for row in csvfile:
            data.append(row)
    return data

def lst_to_dct(lst):
    ''' given a 2d list, create and return a dictionary.
        keys of the dictionary come from the header (first row), 
        values are corresponding columns, saved as lists
    '''
    
    #create dictionary with keys for each column header
    dct = {h : [] for h in lst[0]}
    
    #make the list of data associated with the column key that key's value in the dict
    for row in lst[1:]:
        for i in range(len(row)):
            dct[lst[0][i]].append(row[i])
    return dct

def convert_hrs_to_seconds(time_in_hours):
    ''' given a time in hours format, with the hours, minutes and seconds 
        amounts split by colons (HH:MM:SS), returns the time in seconds as 
        an integer
    '''
    
    #split the time data by colons and turn the hrs, mins, secs values into ints
    split_time = time_in_hours.split(":")
    hours = int(split_time[0])
    minutes = int(split_time[1])
    seconds = int(split_time[2])
    
    #convert the hrs, mins, secs amounts into a seconds sum, returning it
    time_in_seconds = (hours * 3600) + (minutes * 60) + seconds
    return time_in_seconds

def convert_seconds_to_hrs(time_in_seconds):
    ''' given a time in seconds, converts the time into an hours, minutes and
        seconds format ("HH:MM:SS") as a string
    '''
    
    #find the hrs, mins and secs amount from the total secs inputted 
    hours = int(time_in_seconds // 3600)
    remaining_seconds = time_in_seconds % 3600
    mins = int(remaining_seconds // 60)
    remaining_seconds = int(time_in_seconds - (hours * 3600) - (mins * 60))
    
    #control for single digit values to ensure the outputted format is "HH:MM:SS"
    if hours < 10:
        hours = str("0" + str(hours))
    if mins < 10:
        mins = str("0" + str(mins))        
    if remaining_seconds < 10:
        remaining_seconds = str("0" + str(remaining_seconds))
        
    #combine the integer values into a string outputted in "HH:MM:SS" format 
    time_in_hours = str(hours) + ":" + str(mins) + ":" + str(remaining_seconds)
    return time_in_hours

def add_dct_seconds_column(dct):
    ''' given a dictionary of lists that contains a list of times in 
        hours/mins/seconds format, adds a new key/value pairing with the key
        being the TIME_SECONDS_HEADER and the value being the list of the times 
        in seconds format
    '''
    
    #convert every value in the hours time column to seconds and add this as a column
    seconds_time_lst = []
    for time in dct[TIME_HOURS_HEADER]:
        seconds_time_lst.append(convert_hrs_to_seconds(time))
    dct[TIME_SECONDS_HEADER] = seconds_time_lst

def create_year_dct(file_directory, file_name):
    ''' given a file directory and a file name, this function creates a 
        "master" dictionary of dictionaries with keys being the years found in 
        a file's name and the values being the dictionary of lists associated
        with that file/year's data
    '''
    
    #find a list of files located within the inputted directory
    year_dct = {}
    files_lst = os.listdir(file_directory)
    
    #for each file, read the file and convert it to a dictionary of lists
    for file in files_lst:
        file_lst = read_csv(file_directory, file)
        file_dct = lst_to_dct(file_lst)
        
        #find each file's year, or all numerbers in the file's name
        file_year_character_lst = []
        for character in file:
            if character.isnumeric():
                file_year_character_lst.append(character)
        file_year = ""
        for character in file_year_character_lst:
            file_year += character
            
        #add a column to each file's dictionary of time in seconds
        add_dct_seconds_column(file_dct)
        
        #create, return a master year dict matching a file's year to its data dict
        year_dct[int(file_year)] = file_dct
    return year_dct

def find_field_mean(dct_of_years, year, desired_field):
    ''' given a dictionary of years, a year and a desired field within
        the year's dictionary of data, this function will find and return 
        the mean of the desired field for that year
    '''
    
    #find the column data associated with the desired year and field
    field_data = dct_of_years[year][desired_field]
    field_data_num = []
    
    #collect a list of the column's data, find and return the mean of this data
    for num in field_data:
        field_data_num.append(int(num))
    return round(statistics.mean(field_data_num), 0)

def find_field_median(dct_of_years, year, desired_field):
    ''' given a dictionary of years, a year and a desired field within
        the year's dictionary of data, this function will find and return 
        the median of the desired field for that year
    '''
    
    #find the column data associated with the desired year and field
    field_data = dct_of_years[year][desired_field]
    field_data_num = []
    
    #collect a list of the column's data, find and return the median of this data
    for num in field_data:
        field_data_num.append(int(num))
    return round(statistics.median(field_data_num), 0)

def distinct_counter(values, desired_value = ""):
    ''' given a list of values and an optional parameter of a desired value to 
        count either return the count of that desired value in the values list
        or a dictionary with the keys being the unique values and the values
        being the counts of those values in the values list
    '''
    
    #find the distinct values within a list by making the list a set
    unique_values = set(values)
    
    #count the number of times a unique value appears within its list
    unique_values_counter_dct = {}
    for unique_value in unique_values:
        value_count = 0
        for i in range(len(values)):
            if values[i] == unique_value:
                value_count += 1
                
        #make a dictionary with unique value keys and their counts as values
        unique_values_counter_dct[unique_value] = value_count
        
    #logic to return only the count of a desired unique value if specified
    if desired_value:
        return unique_values_counter_dct[desired_value]
    
    #returns the entire dictionary if no desired unique value is specified
    else:
        return unique_values_counter_dct

def find_max(dct_of_years, year, field_name, value_to_exclude = ""):
    ''' given a dictionary of years, a year, a desired field within
        the year's dictionary of data, and an optional value to exlcude from the
        field's values, this function will find the list of the values associated 
        with the desired year and field (excluding the value specified if 
        applicable) and then return the value with the highest frequency in 
        this field 
    '''
    
    #find the column data associated with the desired year and field
    field_data = dct_of_years[year][field_name]
    excl_field_data = []
    
    #determine if each value should be included based on optional exclsuion parameter
    for value in field_data:
        if value != value_to_exclude:
            excl_field_data.append(value)
            
    #use the counter function to create a dictionary with counts of each value
    counter_dct = distinct_counter(excl_field_data)
    
    #find the value with the position of the value with the highest count
    max_position = list(counter_dct.values()).index(
        max(list(counter_dct.values())))
    
    #return the value with the highest count
    return list(counter_dct.keys())[max_position]

def sort_data(dct_of_years, year, desired_field, specifications_field, 
              specification_value):
    ''' given a dictionary of years, a year, a desired field within
        the year's dictionary of data, a second field within that dictionary, 
        and a required value for that second field, this function will find the 
        list of the values associated with the desired year and field and then
        sort those values to only include the values that also have the 
        specified value in the specification field (i.e. will return a list of
        finish times associated only with American runners)
    '''
    
    #find the column data associated with the desired year and field
    desired_field_data = dct_of_years[year][desired_field]
    
    #find the column data associated with the desired year and specified field
    specification_field_data = dct_of_years[year][specifications_field]
    specified_desired_field_data = []
    
    #logic to ensure that all desired field data meets the specified field requirement
    for i in range(len(desired_field_data)):
        if specification_field_data[i] == specification_value:
            specified_desired_field_data.append(desired_field_data[i])
    return specified_desired_field_data
        
def find_yearly_average(years, dct_of_years, desired_field, 
                        specifications_field = "", 
                        specification_value = ""):
    ''' given a list of years, dictionary of years, a desired field within each
        year's dictionary of data, an optional second field within each 
        dictionary, and an optional required value for that second field, 
        this function will find the average of each year's desired field data
        for only the values that aslo have the optional specified value
        in the specification field (i.e. will find the yearly averages of
        finish times associated only with American runners)
    '''
    
    #if there is a specified value, sort yearly field data based off of it
    year_means = []
    for year in years:
        if specifications_field:
            desired_field_data = sort_data(dct_of_years, year, desired_field, 
                                       specifications_field, specification_value)
            
        #if not, attain yearly data associated with desired field 
        else:
            desired_field_data = dct_of_years[year][desired_field]
            
        #find and return the yearly mean for the desired field
        year_means.append(statistics.mean(desired_field_data))    
    return year_means

def find_yearly_median(years, dct_of_years, desired_field):
    ''' given a list of years, a dictionary of years and a desired_field within
        each year's dictionary of data, this function will find the yearly
        median of the desired field for each of the years in the list of years
    '''
    
    #find the data associated with the desired field for each year
    year_medians = []
    for year in years:
        desired_field_data = dct_of_years[year][desired_field]
        
        #find and return as a list the yearly median associated with the field data
        desired_field_data_int = [int(x) for x in desired_field_data]
        year_medians.append(statistics.median(desired_field_data_int))
    return year_medians

def find_years_lst(first_year, last_year, years_excluded = []):
    ''' given a first year, last year, and a list of any years to be excluded, 
        this function will return a list of years that meet those conditions
    '''
    
    #find the list of years from the first year to the last year specified
    every_year_lst = range(first_year, last_year + 1)
    
    #exclude any years specified to be exlcuded, returning this final list
    year_lst = [year for year in every_year_lst if year not in years_excluded]
    return year_lst 

def find_correlation(lst_1, lst_2):
    ''' given two lists of data, if the lists are the same size, the function
        will find and return the correlation between the lists, rounded to 
        4 decimal points
    '''
    
    #ensure both lists are the same size (and thus correlation can be found)
    if len(lst_1) == len(lst_2):
        
        #find and return the correlation, if it can be found
        correlation = round(statistics.correlation(lst_1, lst_2), 4)
        return correlation
    else:
        return "ERROR"

def predict_year_average(lst_1, lst_2, x_value_of_interest):
    ''' given two lists of data, and an x value of interest, this function
        will run a linear regression on these two lists and determine the 
        predicted output associated with the specific x value
    '''
    
    #find the linear regression information for two lists
    linear_regression = stats.linregress(x = lst_1, y = lst_2)
    
    #find, return expected y_value for the desired x_value based off of this linear regression
    year_of_interest_average = linear_regression.intercept + (
        linear_regression.slope * x_value_of_interest)
    return convert_seconds_to_hrs(round(year_of_interest_average, 0))

def normalize_lst(lst):
    ''' given a list of quantitative data, this function will normalize the 
        data (standardizing it around its max and min)
    '''
    
    #normalize each value in the list entered
    normalized_lst = []
    for x in lst:
        normalized_lst.append((int(x) - int(min(lst))) / (int(max(lst)) - 
                                                          int(min(lst))))
   
    #return the normalized list
    return normalized_lst

def plot_time_linear_regression(years, year_averages):
    ''' given a list of years and the average finish times associated with 
        those years, this function graphs a linear regression of the years 
        against the year average finish times – labelling and saving the output
        
    '''
    
    #plit a linear regression plot for the years and year averages lists
    linear_regression_plot = sns.regplot(x = years, y = year_averages, 
                                         color = "red")
    
    #convert y axis finish time in seconds labels to finish times in hours labels
    seconds_y_ticks = [9400, 9600, 9800, 10000, 10200, 10400, 10600]
    hours_y_ticks = [convert_seconds_to_hrs(y) for y in seconds_y_ticks]
    linear_regression_plot.set_yticks(seconds_y_ticks, hours_y_ticks)
    
    #label, show and save the graph
    plt.xlabel("Year")
    plt.ylabel("Average Finish Time")
    plt.title("Year vs Average Finish Time for Americans in the Boston Marathon")
    plt.savefig("hw2_linear_regression.png", bbox_inches = "tight")
    plt.show()
        
def plot_yearly_averages_medians(years, dct_of_years, average_field, median_field):
    ''' given a list of years, a dictionary of years, a desired field to find 
        the yearly averages of, and a desired field to find the yearly median 
        of, this function graphs the yearly average of the average field 
        and the yearly median of the median field over the period of years in
        the list of years entered — labelling and saving the output
    '''
    
    #find the yearly averages and medians associated with the specified fields
    yearly_averages = find_yearly_average(years, dct_of_years, average_field)
    yearly_medians = find_yearly_median(years, dct_of_years, median_field)
    
    #normalize the yearly averages and medians
    normalized_averages = normalize_lst(yearly_averages)
    normalized_medians = normalize_lst(yearly_medians)
    
    #plot on a lineplot these normalized yearly averages and medians
    sns.lineplot(x = years, y = normalized_averages, color = "blue", 
                 label = "Average Finish Time")
    sns.lineplot(x = years, y = normalized_medians, color = "green", 
                 label = "Median Age")
    
    #label, show and save the graph
    plt.xlabel("Year")
    plt.title("Year vs Average Finish Time and Median Age in the Boston Marathon")
    plt.savefig("hw2_normalized_lineplot.png", bbox_inches = "tight")
    plt.show()

def main():
    
    #create the year dct of dct (master dictionary) 
    year_dct = create_year_dct(FILE_DIR, FILE_NAME)
    
    #calculate and print the mean 2013 finish time
    mean_time_2013 = convert_seconds_to_hrs(find_field_mean(year_dct, 2013, 
                                                            TIME_SECONDS_HEADER))
    print(f"The mean finish time for the top 1000 runners in 2013 is {mean_time_2013}")
    
    #calculate and print the median 2010 age
    median_age_2010 = find_field_median(year_dct, 2010, AGE_HEADER)
    print(f"The median age of the top 1000 runners in 2010 is {median_age_2010} years old")
    
    #calculate and print the top country (excluding USA) in 2023
    country_max_2023 = find_max(year_dct, 2023, COUNTRY_RES_ABBR_HEADER, "USA")
    print(f"The country with the most runners in 2023 (apart from the US) is {country_max_2023}")
    
    #calculate and print the number of female runners in the top 1000 in 2021
    female_wins_2021 = distinct_counter(year_dct[2021][GENDER_HEADER], "F")
    print(f"The number of female runners in the top 1000 in 2021 is {female_wins_2021}")
    
    #calcualte and print the correlation between years and female average finish time
    years = find_years_lst(FIRST_YEAR, LAST_YEAR, YEARS_EXCLUDED)
    female_averages = find_yearly_average(years, year_dct, TIME_SECONDS_HEADER, 
                                          GENDER_HEADER, "F")
    female_correlation = find_correlation(years, female_averages)
    print(f"The correlation between year and average finish time for females in the top 1000 runners is {female_correlation}")
    
    #calcualte and print the correlation between years and American average finish time
    american_averages = find_yearly_average(years, year_dct, 
                                            TIME_SECONDS_HEADER, 
                                            COUNTRY_RES_ABBR_HEADER, "USA")
    american_correlation = find_correlation(years, american_averages)
    print(f"The correlation between year and average finish time for Americans in the top 1000 runners is {american_correlation}")
    
    #calculate and print what the expected American average finish time would have been in 2020
    average_american_time_2020 = predict_year_average(years, 
                                                      american_averages, 2020)
    print(f"If the Boston Marathon had happened in 2020, the average finish time for Americans in the top 1000 runners would have been around {average_american_time_2020}")
    
    #plot a linear regression graph of years and American average finish time
    plot_time_linear_regression(years, american_averages)
    
    #plot a normalized line plot of average finish time and median age vs years
    plot_yearly_averages_medians(years, year_dct, TIME_SECONDS_HEADER, 
                                 AGE_HEADER)
    
if __name__ == "__main__":
    main()