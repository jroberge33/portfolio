"""
File: test_assignta.py
Description: use three provided test cases to check the objective functions work

Jack Roberge
March 28th, 2025
"""

import pandas as pd
from assignta import (unpreferred_objective, unavailable_objective, undersupport_objective, overallocation_objective,
                       conflicts_objective)

# create test files
test_1 = pd.read_csv('test1.csv', header=None).to_numpy()
test_2 = pd.read_csv('test2.csv', header=None).to_numpy()
test_3 = pd.read_csv('test3.csv', header=None).to_numpy()

# create dict based on anticipated scores from assignment
test_scores = {'overallocation': [34, 37, 19],
          'conflicts': [7, 5, 2],
          'undersupport': [1, 0, 11],
          'unavailable': [59, 57, 34],
          'unpreferred': [10, 16, 17]}

# TEST FUNCTIONS
def test_overallocation():
    assert overallocation_objective(test_1) == test_scores['overallocation'][0]
    assert overallocation_objective(test_2) == test_scores['overallocation'][1]
    assert overallocation_objective(test_3) == test_scores['overallocation'][2]

def test_conflicts():
    assert conflicts_objective(test_1) == test_scores['conflicts'][0]
    assert conflicts_objective(test_2) == test_scores['conflicts'][1]
    assert conflicts_objective(test_3) == test_scores['conflicts'][2]

def test_undersupport():
    assert undersupport_objective(test_1) == test_scores['undersupport'][0]
    assert undersupport_objective(test_2) == test_scores['undersupport'][1]
    assert undersupport_objective(test_3) == test_scores['undersupport'][2]

def test_unavailable():
    assert unavailable_objective(test_1) == test_scores['unavailable'][0]
    assert unavailable_objective(test_2) == test_scores['unavailable'][1]
    assert unavailable_objective(test_3) == test_scores['unavailable'][2]

def test_unpreferred():
    assert unpreferred_objective(test_1) == test_scores['unpreferred'][0]
    assert unpreferred_objective(test_2) == test_scores['unpreferred'][1]
    assert unpreferred_objective(test_3) == test_scores['unpreferred'][2]