"""
File: evo.py
Description: A concise evolutionary computing framework for solving multi-objective optimization problems

Jack Roberge
March 28th, 2025
"""

import random as rnd
import copy # doing deep copies
from functools import reduce # for discarding dominated (bad) solutions
import time
import csv
from profiler import profile

class Evo:

    def __init__(self):
        """ framework constructor """
        self.pop = {} # population of solutions: evaluation --> solution
        self.fitness = {} # objectives: name --> objective function (f)
        self.agents = {} # agents: name --> (operator/function, num_solutions_input)

    def add_objective(self, name, f):
        """ register a new objective for evaluating solutions """
        self.fitness[name] = f

    def add_agent(self, name, op, k=1):
        """ register an agent take works on k input solutions """
        self.agents[name] = (op, k)

    def get_random_solutions(self, k=1):
        """ Picks k random solutions from the population
        and returns them as a list of deep-copies """
        if len(self.pop) == 0: # no solutions - this shouldn't happen!
            return []
        else:
            solutions = tuple(self.pop.values())
            return [copy.deepcopy(rnd.choice(solutions)) for _ in range(k)]

    def add_solution(self, sol):
        """ Adds the solution to the current population.
        Added solutions are evaluated with respect to each
        registered objective """

        # create the evaluation key
        # key: ((objname1, objvalue1), (objname2, objvalue2), ...)
        eval = tuple([(name, f(sol)) for name, f in self.fitness.items()])

        # Add to the dictionary
        self.pop[eval] = sol

    def run_agent(self, name):
        """ invoking a named agent against the current population """
        op, k = self.agents[name]
        picks = self.get_random_solutions(k)
        new_solution = op(picks)
        self.add_solution(new_solution)

    @staticmethod
    def _reduce_nds(S, p):
        return S - {q for q in S if Evo._dominates(p, q)}

    @staticmethod
    def _dominates(p, q):
        """ p = evaluation of solution: ((obj1, score1), (obj2, score2), ... )"""
        pscores = [score for _, score in p]
        qscores = [score for _, score in q]
        score_diffs = list(map(lambda x, y: y - x, pscores, qscores))
        min_diff = min(score_diffs)
        max_diff = max(score_diffs)
        return min_diff >= 0.0 and max_diff > 0.0

    def remove_dominated(self):
        """ Remove solutions from the pop that are dominated (worse) compared
        to other existing solutions. This is what provides selective pressure
        driving the population towards the pareto optimal tradeoff curve. """
        nds = reduce(Evo._reduce_nds, self.pop.keys(), self.pop.keys())
        self.pop = {k: self.pop[k] for k in nds}

    @ profile
    def evolve(self, time_limit = 1, dom=100, status = 1000):
        """ run the framework (start evolving solutions)
        n = # of random agent invocations (# of generations
        You'll probably need to run tens of thousands of agent invocations
        to produce non-trivial results """

        agent_names = list(self.agents.keys())
        start_time = time.time()
        end_time = start_time + time_limit

        # limit time to ensure < 300 seconds
        while time.time() < end_time:
            pick = rnd.choice(agent_names) # pick an agent to run
            self.run_agent(pick)
            if len(self.pop) % dom == 0:
                self.remove_dominated()

        self.remove_dominated()

    def __str__(self):
        """ Output the solutions in the population """
        rslt = ""
        for eval, sol in self.pop.items():
            rslt += str(dict(eval)) + ":\t" + str(sol) + "\n"
        return rslt


    def to_csv(self, filename='summary.csv'):
        # writes the best solutions to a file with the group name and objective outputs
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['name'] + list(self.fitness.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for eval, sol in self.pop.items():
                row_dict = {'name': 'JR', **dict(eval)}
                writer.writerow(row_dict)

