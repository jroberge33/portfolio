"""
File: assignta.py
Description: main program for assigning tas with agents, objectives, data reading and main function

Jack Roberge
March 28th, 2025
"""

import pandas as pd
import numpy as np
import random as rnd
from evo import Evo
from profiler import profile, Profiler

# read in data
df_ta = pd.read_csv('tas.csv')
df_section = pd.read_csv('sections.csv')

# drop unused info
df_section.drop(['instructor', 'location', 'students', 'topic'], axis=1, inplace=True)
df_preference = df_ta.drop(['ta_id', 'name', 'max_assigned'], axis=1)

# map preferences to numeric values for ease of calculation
preference_map = {"U": 2, "W": 1, "P": 0}
df = df_preference.replace(preference_map)
np_preference = df.to_numpy()

# overallocation data
max_tas = df_ta['max_assigned'].to_numpy()

# undersupport data
min_tas = df_section['min_ta'].to_numpy()

# conflicts data
df_meet_time = df_section['daytime']
meet_times_map = {'R 1145-125': 1, 'W 950-1130': 2, 'W 1145-125': 3,
            'W 250-430': 4, 'W 440-630': 5, 'R 950-1130': 6,
            'R 250-430': 7}
df_meet_time = df_meet_time.replace(meet_times_map)
times = df_meet_time.to_numpy()

# objective functions
@ profile
def overallocation_objective(A):
    """
    overallocation objective for when there are more TAs than allowed.
    :param A: numpy array representing solution with the rows as the TAs and columns as sections
    :return: Count of over-allocations for all TAs
    """
    return sum(np.maximum(np.sum(A, axis=1) - max_tas, 0))

@ profile
def conflicts_objective(A):
    """
    Calculates # of TAs with conflicts
    :param A: numpy array representing solution
    :return: Count of conflicts (overlapping TA assignments)
    """
    # Expands the meet times for each section and makes an array matching solution size
    meet_times = np.tile(times, (len(A), 1))
    # Multiply with the solution to filter out unassigned times
    assigned_times = meet_times * A
    # Creates list of which ta indexes have conflicting time slots
    conflicts = [idx for idx, row in enumerate(assigned_times)
                   if len(set(filter(lambda x: x != 0, row))) != len([x for x in row if x != 0])]
    return len(conflicts)

@ profile
def undersupport_objective(A):
    """
    Counts sections that do not have enough TAs assigned
    :param A: numpy array representing solution
    :return: Amount of understaffing (# of missing TAs)
    """
    return sum(np.maximum(min_tas - np.sum(A, axis=0), 0))

@ profile
def unavailable_objective(A):
    """
    Counts number of assignments where a TA is unavailable
    :param A: numpy array representing solution
    :return: Number of TAs assigned to section they're not available for
    """
    return np.sum(A[np_preference == 2])

@ profile
def unpreferred_objective(A):
    """
    Counts number of assignments where a TA's assignment is not preferred
    :param A: numpy array representing solution
    :return: Number of TAs assigned to section they don't prefer
    """
    return np.sum(A[np_preference == 1])


# agent functions

@ profile
def mix_columns(solutions):
    '''
    Agent that mixes columns from two solutions around randomly
    :param solutions: a list of arrays representing solution
    :return: new solution
    '''
    # get random sols
    sol1, sol2 = rnd.sample(solutions, 2)
    num_cols = sol1.shape[1]
    # create random boolean mask with True = take column from sol1, False = take column from sol2
    mask = np.random.rand(num_cols) < 0.5
    # choose column based on mask
    merged_solution = np.where(mask, sol1, sol2)
    return merged_solution

@ profile
def mix_rows(solutions):
    '''
    Agent that mixes rows from two solutions around randomly
    :param solutions: a list of arrays representing solution
    :return: new solution
    '''
    # get random sols
    sol1, sol2 = rnd.sample(solutions, 2)
    num_rows = sol1.shape[0]
    # create random boolean mask with True = take row from sol1, False = take row from sol2
    mask = np.random.rand(num_rows) < 0.5
    # choose row based on mask
    merged_solution = np.where(mask[:, np.newaxis], sol1, sol2)
    return merged_solution

@ profile
def zero_to_one(solutions):
    """
    Agent: Changes a 0 to 1 (assigns TA)
    :param solutions: a list of arrays representing solution
    :return: new solution
    """
    # get random solution
    solution = rnd.choice(solutions)
    # change random zero to one (assign random TA)
    solution[rnd.randint(0, solution.shape[0] - 1), rnd.randint(0, solution.shape[1] - 1)] = 1
    return solution

@ profile
def one_to_zero(solutions):
    """
    Agent: Changes a 1 to 0 (unassigns TA)
    :param solutions: a list of arrays representing solution
    :return: new solution
    """
    # get random solution
    solution = rnd.choice(solutions)
    # change random one to zero (unassigns TA)
    solution[rnd.randint(0, solution.shape[0] - 1), rnd.randint(0, solution.shape[1] - 1)] = 0
    return solution


# main function

@ profile
def main():

    # initialize base/start schedule
    empty_array = np.zeros((df_preference.shape[0], df_preference.shape[1]))

    # randomly fill array with 1s (assignments)
    L = np.random.randint(0, 2, size=empty_array.shape)

    # create Evo() object
    E = Evo()

    # add all objectives
    E.add_objective("overallocation", overallocation_objective)
    E.add_objective("conflicts", conflicts_objective)
    E.add_objective("undersupport", undersupport_objective)
    E.add_objective("unavailable", unavailable_objective)
    E.add_objective("unpreferred", unpreferred_objective)

    # add all agents
    E.add_agent("merge_columns", mix_columns, 2)
    E.add_agent("merge_rows", mix_rows, 2)
    E.add_agent("assign_ta", zero_to_one, 1)
    E.add_agent("remove_ta", one_to_zero, 1)

    # add the base/start solution
    E.add_solution(L)

    # evolve
    E.evolve(time_limit=300, dom=100)

    # save the best solutions to csv
    E.to_csv(filename = 'JR_EQ_LN_summary.csv')

    # output and save profiler report
    Profiler.report(filename = 'JR_EQ_LN_profile.txt')

main()