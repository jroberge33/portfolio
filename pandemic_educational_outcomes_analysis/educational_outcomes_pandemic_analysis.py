#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jack Roberge
Educational Outcomes Through the Pandemic Analysis 
December 8, 2022
"""

import matplotlib.pyplot as plt

#initializes ten files as constants 
ENROLLMENT_DATA_2019 = "enrollment_data_2019.txt"
ENROLLMENT_DATA_2021 = "enrollment_data_2021.txt"
ATTAINMENT_DATA_2017 = "educational_attainment_data_2017.txt"
ATTAINMENT_DATA_2019 = "educational_attainment_data_2019.txt"
ATTAINMENT_DATA_2021 = "educational_attainment_data_2021.txt"
POPULATION_DATA_2017 = "population_data_2017.txt"
POPULATION_DATA_2019 = "population_data_2019.txt"
POPULATION_DATA_2021 = "population_data_2021.txt"
TEST_SCORES_DATA = "state_test_scores.txt"

def get_labels(file_name, skip_first_row):
    '''
    Obtains and cleans the file's labels as a list
    
    Parameters
    ----------
    file_name : file
        name of the file
    skip_first_row : boolean
        whether or not there's a line we need to skip in the header to make 
        
    Returns
    -------
    cleaned_labels : str
        labels without the headers and split by delimeter and uncecessary spaces
    '''
    
    #opens file and determines if first row should be skipped based off of boolean parameter
    file = open(file_name, "r")
    if skip_first_row == True: 
        file.readline()
        
    #takes in, splits, and cleans labels list
    labels = file.readline()
    cleaned_labels = labels.strip().split(",")
    return cleaned_labels

def read_data(file_name, file_type, skip_first_row):
    '''
    Reads the data into a state-specific list of dictionaries 
    
    Parameters
    ----------
    file_name : file
        name of the file 
    file_type : list
         list of types of data
    skip_first_row : boolean
        whether or not there's a line we need to skip in the header to make 

    Returns
    -------
    state_data : list of dictionaries
        list of state-specific dictionaries with state's specific attributes from the file
    '''
    
    #open file, obtain clean labels list, skip header and empty lines
    file = open(file_name, "r")
    cleaned_labels = get_labels(file_name, skip_first_row)
    file.readline()
    file.readline()
    state_data = []
    
    #clean and split each state's data, creating a dictionary for that state's attributes
    for line in file:
        clean_words = line.strip().split(",")
        state_information = {}
        
        #set equal to zero any empty values in the data sets, indicated as "N"
        for i in range(len(clean_words)): 
            if clean_words[i] != "N":    
                state_information[cleaned_labels[i]] = file_type[i](clean_words[i])
            else:
                state_information[cleaned_labels[i]] = 0
                
        #append individual state's dictionary of attributes to running list of state's dictionaries, close file
        state_data.append(state_information)  
    file.close()
    return state_data

def find_enrollment_change(enrollment_file1, enrollment_data_set_year1, enrollment_data_set_year2):
    '''
    Function to find the percent change between two data sets about state-wide enrollment information
    
    Parameters
    ----------
    enrollment_file1 : file
        name of the file containing enrollment data
    enrollment_data_set_year1 : file
        name of the file containing enrollment data for 2019
    enrollment_data_set_year2 : file
        name of the file containing enrollment data for 2021

    Returns
    -------
    differences : a dictionary of lists
        dict with state as key, list of each enrollment category % change for that state as value
    '''
    
    #get labels of file, control to ignore first two labels: state ID and name
    cleaned_labels = get_labels(enrollment_file1, True)
    data_labels = cleaned_labels[2:12]
    differences = {} 
    
    #iterate through each state in the data set
    for i in range(len(enrollment_data_set_year1)):
        deltas = [] 
        
        #iterate through each enrollment category (nursery, kindergarten, elementary, etc)
        for label in data_labels:
            
            #find percent difference between that state's enrollment category from 2019 to 2021
            difference = enrollment_data_set_year2[i][label] - enrollment_data_set_year1[i][label]
            percent_change = round((difference / enrollment_data_set_year1[i][label]) * 100, 2)
            deltas.append(percent_change)
            
        #differences dictionary with enrollment category as keys, list of state % changes as values
        differences[enrollment_data_set_year1[i][cleaned_labels[1]]] = deltas 
    return differences

def get_data(data_set, desired_column_number):
    '''
    Creates a list of information of specific data from desired columns     
    
    Parameters
    ----------
    data_set : dictionary
        dictionary containing data
    desired_column_number : int
        specified column index in data list

    Returns
    -------
    data_list : list
        list of column-specific information
    '''
    
    #create data list to save column-specific data
    data_list = []
    
    #iterate through data set of interest, appending data from desired column
    for key in data_set:
        data = data_set[key] #gets value matched to desired key 
        data_list.append(data[desired_column_number]) #appends list
    return data_list

def nation_enrollment_averages(enrollment_file1, differences_data):
    '''
    Function to compute enrollment average data across nation 
    
    Parameters
    ----------
    enrollment_file1 : file
        file name 
    differences_data : dictionary of lists
        dict with state as key, list of each enrollment category % change for that state as value

    Returns
    -------
    averages : dictionary
        dictionary of each enrollment category, corresponding average % change
    '''
    
    #get labels of file, control to ignore first two labels: state ID and name
    cleaned_labels = get_labels(enrollment_file1, True)
    data_labels = cleaned_labels[2:12]
    averages = {} 
    
    #iterate through each enrollment category
    for i in range(len(data_labels)):
        
        #produce list of enrollment category's state % changes, find average
        data_list = get_data(differences_data, i)
        data_average = round((sum(data_list) / len(data_list)), 2)
        averages[data_labels[i]] = data_average
    return averages

def create_enrollment_change_bar_graph(averages_data, desired_title, png_title):
    '''
    Function to visualze enrollment change data

    Parameters
    ----------
    averages_data : dictionary
        dictionary of each enrollment category, corresponding average % change
    desired_title : str
        title of  bar graph
    png_title : str
       name of png file that saves the figure

    Returns
    -------
    None.

    '''
    
    #iterates through key-value pairs in the dictionary of averages and plots bar graph
    for key in averages_data: 
        
        #for visual purposes to differentiate growth/decline in enrollment categories
        if averages_data[key] >= 0:
            plt.bar(key, averages_data[key], color = "blue")
        if averages_data[key] < 0:
            plt.bar(key, averages_data[key], color = "red")
    plt.xticks(rotation = 90)
    plt.title(desired_title)
    plt.savefig(png_title + ".png", bbox_inches = "tight")
    plt.show()
    
def user_state_enrollment(state_of_interest, enrollment_file1, differences):
    '''
    Prints and visualizes enrollment data about user-specified state
    
    Parameters
    ----------
    state_of_interest : str
        user input of name of state
    enrollment_file1 : file
        file of 2019 enrollment data
    differences : dictionary
        dict with state as key, list of each enrollment category % change for that state as value
    
    Returns
    -------
    None.

    '''
    
    #get labels of file, control to ignore first two labels: state ID and name
    cleaned_labels = get_labels(enrollment_file1, True)
    data_labels = cleaned_labels[2:12]
    
    #conditional to indicate if user-inputted state is not a real state
    if state_of_interest not in differences:
        print("This is not a valid state.")
    else: 
        
        #index differences data to find enrollment change data for user state
        state_data = differences[state_of_interest]
        
        #output enrollment change for user state, printed and via bar graph
        print()
        print(state_of_interest, "saw the following educational enrollment changes as a result of the pandemic (from 2019 to 2021): ")
        for i in range(len(state_data)):
            print(data_labels[i], "saw a change of", state_data[i], "%")
            
            #for visual purposes to differentiate growth/decline in enrollment categories
            if state_data[i] >= 0:
                plt.bar(data_labels[i], state_data[i], color = "blue")
            if state_data[i] < 0:
                plt.bar(data_labels[i], state_data[i], color = "red")
        plt.xticks(rotation = 90)
        plt.title("School Enrollment Percent Change from 2019 to 2021 for " + state_of_interest)
        plt.savefig("state_enrollment_change.png", bbox_inches = "tight")
        plt.show()
        
def enrollment_data_function(state_of_interest):
    '''
    Performs all enrollment data-related functions

    Parameters
    ----------
    state_of_interest : str
        user-inputted state

    Returns
    -------
    None.

    '''
    
    #list of data types for the enrollment data
    file_type_enrollment = [str, str, int, int, int, int, int, int, int, int, int, int]
    
    #reads through 2019 and 2021 enrollment data producing list of dicts
    enrollment_data_2019 = read_data(ENROLLMENT_DATA_2019, file_type_enrollment, True)
    enrollment_data_2021 = read_data(ENROLLMENT_DATA_2021, file_type_enrollment, True)
    
    #find state enrollment differences, national averages
    pandemic_differences = find_enrollment_change(ENROLLMENT_DATA_2019, enrollment_data_2019, enrollment_data_2021)
    enrollment_averages = nation_enrollment_averages(ENROLLMENT_DATA_2019, pandemic_differences)
    
    #output national, user state enrollment changes
    create_enrollment_change_bar_graph(enrollment_averages, "Average School Enrollment Percent Change from 2019 to 2021", "national_enrollment_change")
    user_state_enrollment(state_of_interest, ENROLLMENT_DATA_2019, pandemic_differences)
    
def get_races(population_file):
    '''
    Function to save the names of different races in a list 

    Parameters
    ----------
    population_file : file
        population file with race demographic data

    Returns
    -------
    races : list
        list of race labels

    '''
    
    #get labels for population/demographic data, control to only include race population labels
    races = [] 
    population_labels = get_labels(population_file, True)
    race_labels = population_labels[2:10]
    
    #find list of races, or the first word of the race population label names
    for race in race_labels:
        words = race.split(" ")
        races.append(words[0])
    return races
    
def find_percent_attainment(attainment_data, population_data, attainment_file, population_file):
    '''    
    Function to update the year's attainment list of dicts to include each state's percent attainment for each race

    Parameters
    ----------
    attainment_data : list of dictionaries
       list of dicts of each state's attainment data
    population_data : dictionaries
        list of dicts with each state's racial population data
    attainment_file : file
        file of attainment data
    population_file : str
        file of population data
        
    Returns
    -------
    None.

    '''
    
    #get list of races from population file
    races = get_races(population_file) 
    
    #iterate through each state, and within each state, each race    
    for i in range(len(attainment_data)):
        for race in races:
            
            #disclude races state has no data on
            if (attainment_data[i][race + " High School"] != 0) and (population_data[i][race + " Population"] != 0):
                
                #add percent attainment of each race to state dict, dividing population of that race with degree by total race population
                attainment_data[i][(race + " High School %")] = round((attainment_data[i][race + " High School"] / population_data[i][race + " Population"]) * 100, 2)
            if (attainment_data[i][race + " Bachelor’s Degree"] != 0) and (population_data[i][race + " Population"] != 0):    
                attainment_data[i][(race + " Bachelor’s Degree %")] = round((attainment_data[i][race + " Bachelor’s Degree"] / population_data[i][race + " Population"]) * 100, 2)
  
def race_attainment_averages(attainment_data, population_file):
    '''
    Calculates attainment averages by race
    
    Parameters
    ----------
    attainment_data : list of dictionaries
       list of dicts of each state's attainment data
    population_file : file
        file of population data
        
    Returns
    -------
    averages : dictionary 
        dict matching race and degree to that race's national degree attainment average

    '''
    
    #get list of races from population file, create averages dict
    races = get_races(population_file)
    averages = {} 
    
    #iterate through each race, and for each race, each state
    for race in races:
        values_list = [] 
        for state in attainment_data:
            
            #control to ensure desired race is reported by state
            if (race + " High School %") in state:
                state_value = state[race + " High School %"]
                
                #appending the list of that race's attainment values
                values_list.append(state_value) 
        
        #calculate attainment average for that race, adding as a key/value pair to averages dict    
        averages[race + " High School %"] = round((sum(values_list) / len(values_list)), 2) 
        
        #repeat for bachelors degree
        values_list = []
        for state in attainment_data:
            if (race + " Bachelor’s Degree %") in state:
                state_value = state[race + " Bachelor’s Degree %"]
                values_list.append(state_value)
        averages[race + " Bachelor’s Degree %"] = round((sum(values_list) / len(values_list)), 2)
    return averages
    
def create_attainment_bar_graph(population_file, attainment_averages, year):
    '''
    Creates bar graph of attainment data for each race

    Parameters
    ----------
    population_file : file
       file of population data
    attainment_averages : dictionary
        dict matching race and degree to that race's national degree attainment average
    year : str
        desired year

    Returns
    -------
    None.

    '''
    
    #get list of races from population file
    races = get_races(population_file) 
    
    #iterate through each race to plot national avg high school degree attainment by race
    for race in races:
        plt.bar(race + " High School %", attainment_averages[race + " High School %"])
        print(race, " High School %: ", attainment_averages[race + " High School %"])
    plt.xticks(rotation = 90)
    plt.title("High School Degree Educational Attainment By Race, " + year)
    plt.savefig("race_highschool_degree.png", bbox_inches = "tight")
    plt.show()
    
    #repeat for bachelor's degree 
    for race in races:
        plt.bar(race + " Bachelor’s Degree %", attainment_averages[race + " Bachelor’s Degree %"])
        print(race, "")
    plt.xticks(rotation = 90)
    plt.title("Bachelor’s Degree Degree Educational Attainment By Race, " + year)
    plt.savefig("race_bachelors_degree.png", bbox_inches = "tight")
    plt.show()

def find_attainment_change(population_file, attainment_data_year1, attainment_data_year2):
    '''
    Calculates percent change in percentage attainment by race for high school, bachelor's degree

    Parameters
    ----------
    population_file : file
       file of population data
    attainment_data_year1 : list of dictionaries
       list of dicts of each state's attainment data in first year of interest
    attainment_data_year2 : list of dictionaries
       list of dicts of each state's attainment data in second year of interest

    Returns
    -------
    differences : dictionary of lists
        dictionary matching race and degree to a list of each state's percent change in percent attainment

    '''
    #get list of races from population file, initialize dict to save differences
    races = get_races(population_file)
    differences = {} 
    for race in races:
        
        #create lists to capture each race's change in high school, bachelors percent attainment
        deltas_highschool = [] 
        deltas_bachelors = [] 
        
        #iterate through each race, iterating through each state within each race
        for i in range(len(attainment_data_year1)):
            
            #control to check if race data is in the state's data for both years
            if (race + " High School %") in attainment_data_year1[i] and (race + " High School %") in attainment_data_year2[i]:              
               
                #calculates absolute difference in percent attainment between two years
                difference_highschool = attainment_data_year2[i][race + " High School %"] - attainment_data_year1[i][race + " High School %"]
                
                #calculate percent change from year 1 to year 2 in percent attainment, add to list of change
                percent_change_highschool = round(difference_highschool / attainment_data_year1[i][race + " High School %"] * 100, 2)
                deltas_highschool.append(percent_change_highschool)
            
            #repeat for bachelors degree
            if (race + " Bachelor’s Degree %") in attainment_data_year1[i] and (race + " Bachelor’s Degree %") in attainment_data_year2[i]:
                difference_bachelors = attainment_data_year2[i][race + " Bachelor’s Degree %"] - attainment_data_year1[i][race + " Bachelor’s Degree %"]
                percent_change_bachelors = round(difference_bachelors / attainment_data_year1[i][race + " Bachelor’s Degree %"] * 100, 2)
                deltas_bachelors.append(percent_change_bachelors) 
                
            #match race and degree to the list of each state's percent change in percent attainment
            differences[race + " High School %"] = deltas_highschool
            differences[race + " Bachelor’s Degree %"] = deltas_bachelors  
    return differences

def attainment_change_averages(population_file, differences_data):
    '''   
    Function to calculate average percent change in percent attainment by race and degree
    
    Parameters
    ----------
    population_file : file
        file of population data
    differences_data : dictionary of lists
        dictionary matching race and degree to a list of each state's percent change in percent attainment

    Returns
    -------
    averages : dictionary
        dictionary matching race and degree to national average percent change in percent attainment 

    '''
    
    #get list of races from population file, initialize dict to save averages
    races = get_races(population_file)
    averages = {}
    
    #iterates through each race, finding national bachelor and high school percent attainment average change
    for race in races:
        averages[race + " High School %"] = round(sum(differences_data[race + " High School %"]) / len(differences_data[race + " High School %"]), 2)
        averages[race + " Bachelor’s Degree %"] = round(sum(differences_data[race + " Bachelor’s Degree %"]) / len(differences_data[race + " Bachelor’s Degree %"]), 2)
    return averages


def create_attainment_change_bar_graph(population_file, averages, year1, year2):
    '''
    creates visualization for attainment change data

    Parameters
    ----------
    population_file : file
        file of population data 
    averages : dictionary
        dictionary matching race and degree to national average percent change in percent attainment 
    year1 : str
       first year
    year2 : str
        second year

    Returns
    -------
    None.

    '''
    
    #get list of races from population file
    races = get_races(population_file)
        
    #iterate through graphing % change in percent high school attainment for every race
    for race in races:
        
        #for visual purposes to differentiate growth/decline in race percent attainment
        if averages[race + " High School %"] >= 0:
            plt.bar(race + " High School %", averages[race + " High School %"], color = "blue")
        if averages[race + " High School %"] < 0:
            plt.bar(race + " High School %", averages[race + " High School %"], color = "red")
    plt.xticks(rotation = 90)
    plt.title("Change in High School Degree Attainment, " + year1 + " - " + year2)
    plt.savefig("national_highschool_degree_change_" + year1 + "_" + year2 + ".png", bbox_inches = "tight")
    plt.show()
    
    #repeat for bachelor's degree
    for race in races:
        if averages[race + " Bachelor’s Degree %"] >= 0:
            plt.bar(race + " Bachelor’s Degree %", averages[race + " Bachelor’s Degree %"], color = "blue")
        if averages[race + " Bachelor’s Degree %"] < 0:
            plt.bar(race + " Bachelor’s Degree %", averages[race + " Bachelor’s Degree %"], color = "red")
    plt.xticks(rotation = 90)
    plt.title("Change in Bachelor's Degree Attainment, " + year1 + " - " + year2)
    plt.savefig("national_bachelors_degree_change_" + year1 + "_" + year2 + ".png", bbox_inches = "tight")
    plt.show()
    
def user_state_attainment(state_of_interest, population_file, attainment_data_file1, attainment_data_file2):
    '''    
    Prints and visualizes attainment information for user state

    Parameters
    ----------
    state_of_interest : str
        user-inputted state
    population_file : file
        file of population data
    attainment_data_file1 : list of dictionaries
       list of dicts of each state's attainment data in first year of interest
    attainment_data_file2 : list of dictionaries
       list of dicts of each state's attainment data in second year of interest

    Returns
    -------
    None.
    
    '''
    
    #get list of races from population file, initialize list of all states
    races = get_races(population_file)
    states = [] 
    
    #for each state in attainment data, add the state's name to the list of states
    for line in attainment_data_file1:
        states.append(line["Geographic Area Name"])
        
    #control to ensure user state is a valid state, find state's index position in attainment data lists
    if state_of_interest in states: 
        state_location = states.index(state_of_interest)
        
        #output pre-pandemic degree percent attainment by race
        print()
        print("Before the pandemic, your state saw the following attainment percentages by race: ")
        print()
        for race in races:
            
            #control to ensure state has data on race
            if (race + " High School %") in attainment_data_file2[state_location]:
                print(race + " High School %:", attainment_data_file2[state_location][race + " High School %"], "%")
            if (race + " Bachelor’s Degree %") in attainment_data_file2[state_location]:
                print(race + " Bachelor’s Degree %:", attainment_data_file2[state_location][race + " Bachelor’s Degree %"], "%")
                
        #output post pandemic changes in state percent attainment by race
        print()
        print("After the pandemic, your state saw the following percent change in each race's educational attainment: ")
        print()
        
        for race in races:
            
            #control to ensure state has both pre and post-pandemic data on race
            if (race + " High School %") in attainment_data_file1[state_location] and (race + " High School %") in attainment_data_file2[state_location]:
                
                #calculate the absolute difference in percent attainment from the first to second year 
                difference_highschool = attainment_data_file2[state_location][race + " High School %"] - attainment_data_file1[state_location][race + " High School %"]
                
                #find the percent change in percent attainment between the years, outputting this value
                percent_change_highschool = round(difference_highschool / attainment_data_file1[state_location][race + " High School %"] * 100, 2)
                print(race + " High School %:", percent_change_highschool, "%")
                
            #repeat for bachelor's degree
            if (race + " Bachelor’s Degree %") in attainment_data_file1[state_location] and (race + " Bachelor’s Degree %") in attainment_data_file2[state_location]:
                difference_bachelors = attainment_data_file2[state_location][race + " Bachelor’s Degree %"] - attainment_data_file1[state_location][race + " Bachelor’s Degree %"]
                percent_change_bachelors = round(difference_bachelors / attainment_data_file1[state_location][race + " Bachelor’s Degree %"] * 100, 2)
                print(race + " Bachelor’s Degree %:", percent_change_bachelors, "%")
       
        #graph post-pandemic percent attainment by race for user state, high school
        for race in races:
            if (race + " High School %") in attainment_data_file2[state_location]:
                plt.bar(race + " High School %", attainment_data_file2[state_location][race + " High School %"])
        plt.xticks(rotation = 90)
        plt.title("High School Degree Attainment By Race in " + state_of_interest + ", 2021")
        plt.savefig("state_highschool_degree_attainment.png", bbox_inches = "tight")
        plt.show()
        
        #repeat for bachelor's degree
        for race in races:
            if (race + " Bachelor’s Degree %") in attainment_data_file2[state_location]:
                plt.bar(race + " Bachelor’s Degree %", attainment_data_file2[state_location][race + " Bachelor’s Degree %"])
        plt.xticks(rotation = 90)
        plt.title("Bachelor's Degree Attainment By Race in " + state_of_interest + ", 2021")
        plt.savefig("state_bachelors_degree_attainment.png", bbox_inches = "tight")
        plt.show()
        
        #for five largest races, produce % change in percent attainment by race graph
        for race in races[0:4]:
            
            #control to ensure state has both pre and post-pandemic data on race
            if (race + " High School %") in attainment_data_file1[state_location] and (race + " High School %") in attainment_data_file2[state_location]:
                
                #calculate the absolute difference in percent attainment from the first to second year 
                difference_highschool = attainment_data_file2[state_location][race + " High School %"] - attainment_data_file1[state_location][race + " High School %"]
                
                #find the percent change in percent attainment between the years, graphing this value
                percent_change_highschool = round(difference_highschool / attainment_data_file1[state_location][race + " High School %"] * 100, 2)
                
                #for visual purposes to differentiate growth/decline in race percent attainment
                if percent_change_highschool >= 0:
                    plt.bar(race + " High School %", percent_change_highschool, color = "blue")
                if percent_change_highschool < 0:
                    plt.bar(race + " High School %", percent_change_highschool, color = "red")                    
        plt.xticks(rotation = 90)
        plt.title("High School Degree Attainment % Change in " + state_of_interest + " from 2019 to 2021")
        plt.savefig("state_highschool_degree_attainment_change.png", bbox_inches = "tight")
        plt.show()
        
        #repeat for bachelor's degree
        for race in races[0:4]:
            if (race + " Bachelor’s Degree %") in attainment_data_file1[state_location] and (race + " Bachelor’s Degree %") in attainment_data_file2[state_location]:
                difference_bachelors = attainment_data_file2[state_location][race + " Bachelor’s Degree %"] - attainment_data_file1[state_location][race + " Bachelor’s Degree %"]
                percent_change_bachelors = round(difference_bachelors / attainment_data_file1[state_location][race + " Bachelor’s Degree %"] * 100, 2)
                if percent_change_bachelors >= 0:    
                    plt.bar(race + " Bachelor’s Degree %", percent_change_bachelors, color = "blue")
                if percent_change_bachelors < 0:
                    plt.bar(race + " Bachelor’s Degree %", percent_change_bachelors, color = "red")                    
        plt.xticks(rotation = 90)
        plt.title("Bachelor's Degree Attainment % Change in " + state_of_interest + " from 2019 to 2021")
        plt.savefig("state_bachelors_degree_attainment_change.png", bbox_inches = "tight")
        plt.show()
    else:
        print("This is not a valid state.")
    
def attainment_data_function(state_of_interest):
    '''
    Performs all attainment data-related functions

    Parameters
    ----------
    state_of_interest : str
        user-inputted state

    Returns
    -------
    None.

    '''
    
    #list of data types for the attainment data
    file_type_attainment_2017 = [str, str, int, int, int, int, int, int, int, int]
    file_type_attainment_2019 = [str, str, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int]
    file_type_attainment_2021 = [str, str, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int]
    file_type_population_2017 = [str, str, int, int, int, int]
    file_type_population_2019 = [str, str, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int]
    file_type_population_2021 = [str, str, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int]
   
    #read through attainment and population data for 2017, 2019, 2021
    attainment_data_2017 = read_data(ATTAINMENT_DATA_2017, file_type_attainment_2017, True)
    attainment_data_2019 = read_data(ATTAINMENT_DATA_2019, file_type_attainment_2019, True)
    attainment_data_2021 = read_data(ATTAINMENT_DATA_2021, file_type_attainment_2021, True)   
    population_data_2017 = read_data(POPULATION_DATA_2017, file_type_population_2017, True)
    population_data_2019 = read_data(POPULATION_DATA_2019, file_type_population_2019, True)
    population_data_2021 = read_data(POPULATION_DATA_2021, file_type_population_2021, True)
    
    #update 2017, 2019, 2021 attainment data with percent attainment by race
    find_percent_attainment(attainment_data_2017, population_data_2017, ATTAINMENT_DATA_2017, POPULATION_DATA_2017)
    find_percent_attainment(attainment_data_2019, population_data_2019, ATTAINMENT_DATA_2019, POPULATION_DATA_2019)
    find_percent_attainment(attainment_data_2021, population_data_2021, ATTAINMENT_DATA_2021, POPULATION_DATA_2021)
    
    #find and graph most recent national percent attainment averages
    attainment_averages_2021 = race_attainment_averages(attainment_data_2021, POPULATION_DATA_2021)
    create_attainment_bar_graph(POPULATION_DATA_2021, attainment_averages_2021, "2021")
    
    #find and graph the percent change in percent attainment from 2017 to 2019, and 2019 to 2021
    differences_2017_2019 = find_attainment_change(POPULATION_DATA_2017, attainment_data_2017, attainment_data_2019)
    differences_2019_2021 = find_attainment_change(POPULATION_DATA_2017, attainment_data_2019, attainment_data_2021)
    averages_2017_2019 = attainment_change_averages(POPULATION_DATA_2017, differences_2017_2019)
    averages_2019_2021 = attainment_change_averages(POPULATION_DATA_2017, differences_2019_2021)
    create_attainment_change_bar_graph(POPULATION_DATA_2017, averages_2017_2019, "2017", "2019")
    create_attainment_change_bar_graph(POPULATION_DATA_2017, averages_2019_2021, "2019", "2021")
    
    #print and graph attainment data for user state
    user_state_attainment(state_of_interest, POPULATION_DATA_2019, attainment_data_2019, attainment_data_2021)

def find_score_labels(test_scores_file):
    '''    
    Function to get labels of test scores subject and grade (Reading-4th, Reading-8th, etc)

    Parameters
    ----------
    test_scores_file : file
        file of test scores data from 2019 and 2022

    Returns
    -------
    yearless_labels : list
        list of labels of the test scores subject and grade

    '''
    
    #get list of clean labels from test scores file
    labels = get_labels(test_scores_file, False)
    
    #control to get rid of the state name from label list, initialize yearless labels list
    labels = labels[1:8]
    yearless_labels = []
    
    #create new list yearless_labels that takes only the subject and grade, no year of the label
    for label in labels:
        split_label = label.strip().split(" ")
        
        #take label first word, like Reading-4th and append to list of yearless labels
        yearless_label = split_label[0]
        yearless_labels.append(yearless_label)
        
    #get rid of repeats, as Reading-4th 2019 and Reading-4th 2022 will produce two identical yearless labels
    yearless_labels = [yearless_labels[0], yearless_labels[2], yearless_labels[4], yearless_labels[6]]  
    return yearless_labels

def find_score_change_average(test_scores_file, test_scores_data):
    '''
    Function to find the average national change in test scores

    Parameters
    ----------
    test_scores_file : file
        file of test score data from 2019 and 2022 
    test_scores_data : list of dictionaries
        list of dicts of each state's test score data

    Returns
    -------
    averages : dictionary
       dictionary storing each subject and grade's average change in test scores post-pandemic

    '''
    
    #get list of yearless labels of score subject and grades, initialize averages dict
    yearless_labels = find_score_labels(test_scores_file)
    averages = {} 
    
    #iterate through for each state, and within each state, for each grade and subject label
    for i in range(len(test_scores_data)):
        for label in yearless_labels:
            
            #computes percent change in test scores from 2019 to 2022
            test_scores_data[i][label + " Percent Change"] = round(((test_scores_data[i][label + " 2022"] - test_scores_data[i][label + " 2019"]) / test_scores_data[i][label + " 2019"]) * 100, 2)
            
    #iterate through for each grade/subject label, initialize list of all state data for each label
    for label in yearless_labels:
        averages_list = [] 
        
        #append each state's average change in each grade/subject scores category
        for state in test_scores_data:
            averages_list.append(state[label + " Percent Change"])
        averages[label] = round(sum(averages_list) / len(averages_list), 2) #computes average
    return averages

def create_score_change_bar_graph(averages):
    '''
    Function to visualize score change data

    Parameters
    ----------
    averages : dictionary
       dictionary storing each subject and grade's average change in test scores post-pandemic

    Returns
    -------
    None.

    '''
    
    #plot the average national change in test scores for each grade/subject scores category
    for key in averages:

        #for visual purposes to differentiate growth/decline in test scores
        if averages[key] >= 0:    
            plt.bar(key + " Grade", averages[key], color = "blue")
        if averages[key] < 0:
            plt.bar(key + " Grade", averages[key], color = "red")
    plt.xticks(rotation = 90)
    plt.title("National Average Change in Test Scores")
    plt.savefig("national_test_scores.png", bbox_inches = "tight")
    plt.show()
    
def user_state_scores(state_of_interest, test_scores_file, test_scores_data):
    '''    
    Print and graph user state test score information

    Parameters
    ----------
    state_of_interest : str
        name of specified state.
    test_scores_file : file
        file of test scores data
    test_scores_data : list of dictionaries
        list of dicts of each state's test score data

    Returns
    -------
    None.

    '''
    
    #get list of yearless labels of score subject and grades, initialize list of state names
    yearless_labels = find_score_labels(test_scores_file)
    states = []

    #append each state's name in test scores data to list of state names
    for line in test_scores_data:
        states.append(line["State"])

    #control to ensure user state is valid state, find state's index position in test scores data
    if state_of_interest in states:  
        state_location = states.index(state_of_interest)
        print()
        
        #print and graph enrollment change for user state of interest
        for label in yearless_labels:
            print(state_of_interest + " saw a change of", test_scores_data[state_location][label + " Percent Change"], "in", label, "Grade test scores.")
            
            #for visual purposes to differentiate growth/decline in test scores
            if test_scores_data[state_location][label + " Percent Change"] >= 0:
                plt.bar(label + " Grade", test_scores_data[state_location][label + " Percent Change"], color = "blue")
            if test_scores_data[state_location][label + " Percent Change"] < 0:
                plt.bar(label + " Grade", test_scores_data[state_location][label + " Percent Change"], color = "red")
        plt.xticks(rotation = 90)
        plt.title("Change in Test Scores for " + state_of_interest)
        plt.savefig("state_test_scores.png", bbox_inches = "tight")
        plt.show()     
    else:
        print("This is not a valid state.")
        
def test_scores_data_function(state_of_interest):
    '''
    Performs all test score data-related functions

    Parameters
    ----------
    state_of_interest : str
       user-inputted state

    Returns
    -------
    None.

    '''
    
    #list of data types for the test scores data
    file_type_scores_data = [str, int, int, int, int, int, int, int, int]
    
    #read and clean test scores data into list of dictionaries for each state's test scores data
    test_scores_data = read_data(TEST_SCORES_DATA, file_type_scores_data, False)
    
    #find and graph average change in test scores pre and post pandemic
    averages = find_score_change_average(TEST_SCORES_DATA, test_scores_data)
    create_score_change_bar_graph(averages)
    
    #print and graph test score change data for user state
    user_state_scores(state_of_interest, TEST_SCORES_DATA, test_scores_data)

def main():
    
    #determine user state of interest
    state_of_interest = input("Please enter your desired state of interest: ")
    
    #format titling and output of each educational metric function
    print()
    print()
    print("SCHOOL ENROLLMENT DATA")
    print()
    enrollment_data_function(state_of_interest)
    print()
    print("EDUCATIONAL ATTAINMENT DATA")
    print()
    attainment_data_function(state_of_interest)
    print()
    print("TEST SCORES DATA")
    print()
    test_scores_data_function(state_of_interest)
    print()
    
if __name__ == "__main__":
    main()
