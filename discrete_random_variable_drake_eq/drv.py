#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jack Roberge
March 9, 2025
"""

# import libraries
import copy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class DRV:

    def __init__(self, dist=None, **kwargs):
        """ Constructor
         dist: Dictionary of value:probability pairs
         kwargs: misc parameters for other types of distributions """

        if dist is None:
            self.dist = dict()  # Empty distribution
        else:
            self.dist = copy.deepcopy(dist)

        # get data type
        dtype = kwargs.get('type', 'discrete')

        # creation of discrete distribution for uniform distribution
        if dtype == 'uniform':

            # get min, max, # of bins
            minval = kwargs.get('min', 0.0)
            maxval = kwargs.get('max', 1.0)
            bins = kwargs.get('bins', 10)

            # creates array of equally sized bins between min and max
            bin_edges = np.linspace(minval, maxval, bins + 1)

            # calculates center/mid-points of bins
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # create discrete distribution with equal probability for each bin
            self.dist.update({round(i, 5): (1/bins) for i in bin_centers})

        # creation of discrete distribution for normal distribution
        elif dtype == 'normal':

            # get mean, stdev and bins
            mean = kwargs.get('mean', 0.0)
            stdev = kwargs.get('stdev', 1.0)
            bins = kwargs.get('bins', 10)

            # create dist range (+/- 3 stdevs represents ~99.7% of data)
            minval = mean - 3 * stdev
            maxval = mean + 3 * stdev

            # calculate bin width and edges of bins
            bin_width = (maxval - minval) / bins
            bin_edges = [minval + i * bin_width for i in range(bins + 1)]

            # create huge normal distribution sample so law of large numbers applies
            samples = np.random.normal(mean, stdev, 1000000)

            # calculate number of samples in each bin, divide by total # for bin probability
            for left_edge, right_edge in zip(bin_edges[:-1], bin_edges[1:]):
                midpoint = (right_edge + left_edge) / 2
                prob = (np.sum((samples >= left_edge) & (samples <= right_edge))) / len(samples)
                self.dist[midpoint] = prob

    def __getitem__(self, x):
        """ Fetching the probability associated with value x"""
        return self.dist.get(x, 0.0)

    def __setitem__(self, x, p):
        self.dist[x] = p

    def E(self):
        """ Expected value E[X] """
        ev = 0.0
        for x, p in self.dist.items():
            ev = ev + x * p
        return round(ev, 5)

    def apply(self, other, op):
        """ Apply an operator function (op) to the 'other' drv """
        Z = DRV()
        for x, px in self.dist.items(): # value:probability pairs of this object
            for y, py in other.dist.items(): # value:probability pairs of the other object
                Z[op(x, y)] = Z[op(x, y)] + px * py
        return Z

    def __add__(self, other):
        """ Add two discrete random variables Z = X + Y """
        # checks if other is a scalar
        if isinstance(other, (int, float)):
            return self.__radd__(other)
        return self.apply(other, lambda x, y: x + y)

    def __sub__(self, other):
        """ Subtract two discrete random variables  """
        # checks if other is a scalar
        if isinstance(other, (int, float)):
            # despite subtraction not commutative this still always returns (self - scalar)
            return self.__rsub__(other)
        return self.apply(other, lambda x, y: x - y)

    def __mul__(self, other):
        """ Multiply two discrete random variables  """
        # checks if other is a scalar
        if isinstance(other, (int, float)):
            return self.__rmul__(other)
        return self.apply(other, lambda x, y: x * y)

    def __radd__(self, a):
        """ Add a scalar, a, by the DRV """
        Z = DRV()
        for x, px in self.dist.items():
            Z[x + a] = px
        return Z

    def __rsub__(self, a):
        """ Subtract scalar - drv """
        Z = DRV()
        for x, px in self.dist.items():
            Z[x - a] = px
        return Z

    def __rmul__(self, a):
        """ Multiply a scalar, a, by the DRV """
        Z = DRV()
        for x, px in self.dist.items():
            Z[x * a] = px
        return Z

    def __repr__(self):
        """ Human-readable string representation of the DRV
        Display each value:probability pair on a separate line.
        Round all probabilities to 5 decimal places. """
        return "\n".join(f"The value {x} has a probability of {px:.5f}" for x, px in self.dist.items())

    def plot(self, title=None, xscale=None, yscale=None, show_cumulative=False,
             savefig=None, figsize=(4, 4)):
        """
        Display the DRV distribution
        title: The title of the figure
        xscale: If 'log' then log-scale the x axis
        yscale: If 'log' then log-scale the y axis
        show_cummulative: If True, overlay the cummulative distribution line
        savefig: Name of .png file to save plot
        figsize: Default figure size
        """

        # get x (variable) and y (probability) values from dist
        x, y = zip(*sorted(self.dist.items()))
        x = np.array(x)

        # create bar plot for probability distribution
        plt.figure(figsize=figsize)
        sns.barplot(x=x, y=y, color='blue', alpha=0.7, label='Probability Distribution')

        # calculate and plot cumulative probability if specified
        if show_cumulative:
            cum_y = np.cumsum(y)
            sns.lineplot(x=x, y=cum_y, marker='o', color='red', label='Cumulative Distribution')

        # add label, legend and title (if specified)
        plt.xlabel("Variable")
        plt.ylabel("Probability")
        if title:
            plt.title(title)
        plt.legend(loc='best')

        # convert to log scale (if specified)
        if xscale == 'log':
            plt.xscale('log')
        if yscale == 'log':
            plt.yscale('log')

        # save figure (if specified) and show plot
        if savefig:
            plt.savefig(savefig, bbox_inches="tight")
        plt.show()






