"""Main script for Polly algorithm."""

import itertools
import math
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

from oracle import ExampleOracle


def graph(formula, x):
    """Plots a 2D function on the x-y plane."""
    y = formula(x)
    plt.plot(x, y)


def midpoint(a, b):
    """Finds the midpoint of two points in n-dimensional space."""
    try:
        return (a + b) / 2
    except:
        return [(i + j) / 2 for i, j in zip(a, b)]


def generate_dataset(dim, num_hyperplanes, eps, delta,
                     dist, oracle, dataset_size=None):
    """Generates a dataset of labeled points."""

    # Calculates the dataset size necessary for theoretical guarantees.
    term1 = 2 / eps * math.log(4 / delta, 2)
    term2 = 16 * (dim + 1) * num_hyperplanes * math.log(3 * num_hyperplanes, 2)
    term2 /= eps
    term2 *= math.log(13 / eps, 2)
    num_samples = math.ceil(max(term1, term2))

    # Overrides num_samples if desired.
    if dataset_size:
        num_samples = dataset_size

    # Samples the calculated number of samples from the given distribution.
    if dist == "uniform":
        dataset = np.random.random_sample((num_samples, dim))
    else:
        # Other distributions.
        return NotImplementedError

    # Uses the oracle to label the dataset.
    labels = np.array([oracle.label_point(p) for p in dataset])

    return dataset, labels


def find_surrogate(point, center, d2_hat, oracle):
    """Finds a surrogate point.

       Given a negatively sampled point and a center guaranteed to be in the
       polytope, uses binary search and oracle queries to find a surrogate point
       which is a maximum distance of d2_hat outside the polytope.
    """

    # Gets the point -> center vector of magnitude d2_hat.
    vec = center - point
    vec /= np.linalg.norm(vec)
    d2_hat_vec = d2_hat * vec

    # Base case for surrogate point is just the midpoint.
    surrogate = midpoint(point, center)

    # Performs binary search to find the surrogate point.
    # Stops the binary search when the surrogate point is less than d2_hat away
    # from the polytope. In other words, the surrogate point is outside the
    # polytope, and the surrogate point + d2_hat_vec is inside the polytope.
    lo = point
    hi = center
    while oracle.label_point(surrogate) or not oracle.label_point(surrogate + d2_hat_vec):
        old_surrogate = surrogate
        # Surrogate point is in the polytope, so move towards the point.
        if oracle.label_point(surrogate):
            surrogate = midpoint(surrogate, lo)
            hi = old_surrogate
        # Surrogate point is outside the polytope, so move towards the center.
        else:
            surrogate = midpoint(surrogate, hi)
            lo = old_surrogate

    return surrogate


def find_separating_hyperplane(set1, set2):
    """Uses linear programming to separate set1 and set2.

       set1 and set2 are assumed to be linearly separable.

       max 0
       subject to:
       -s * x + b <= -1 for all s in set1
       s * x - b <= -1 for all s in set2
    """

    c = [0] * (dim + 1)

    A = []
    for s in set1:
        A.append(list(-s) + [1])
    for s in set2:
        A.append(list(s) + [-1])

    b = [-1] * (len(set1) + len(set2))

    lp = linprog(c=c, A_ub=A, b_ub=b, bounds=[-np.inf, np.inf])

    return lp["x"], lp["success"]


def plot2d(hyperplanes, dataset):
    """Plots 2D results."""

    # Plots hyperplanes.
    for h in hyperplanes:
        a1 = h[0][0]
        a2 = h[0][1]
        b = h[1]

        m = -a1 / a2
        b = b / a2
        graph(lambda x: m*x+b, np.arange(0, 1, 0.01))

    # Plots dataset.
    for d in dataset:
        if oracle.label_point(d):
            plt.plot(d[0], d[1], marker='o', markersize=1, color="red")
        else:
            plt.plot(d[0], d[1], marker='o', markersize=1, color="blue")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.savefig("pred.png")


def polly(dim, num_hyperplanes, dataset, labels, oracle):
    # Guesses at maximum distance of surrogate points to concept boundary.
    d2_hat = math.sqrt(dim) / (24 * len(dataset) * math.pow(num_hyperplanes, 2))
    d2_hat *= math.pow((1 / (math.pow(dim, 1.5) * len(dataset))), dim)

    # d2_hat above is important for theoretical guarantees, but it is often
    # extremely low and explodes the runtime of finding the surrogate points.
    if d2_hat < 1e-8:
        print("Original d2_hat: {}".format(d2_hat))
        print("Too small for computation.")
        print("New d2_hat: 1e-8\n")
        d2_hat = 1e-8

    # Divides the dataset into positive and negative labeled samples.
    positive_samples = np.array(dataset[labels])
    negative_samples = np.array(dataset[~labels])

    pm = positive_samples.mean()

    # d is the minimum distance between the polytope and any negative sample.
    # We cannot get it exactly, so we calculate a lower bound.
    # We assume a bit complexity of 10, but this can be changed.
    bit_complexity = 10
    d = 1 / (math.pow(2, bit_complexity) * math.sqrt(dim))
    step = 2 * d / (len(dataset) * math.sqrt(dim))
    step /= 10
    r = range(-5, 6)

    # We choose surrogate points relative to a "center"; that is, some
    # point randomly selected from a ball lying completely inside the polytope.
    ball = []
    for x in itertools.product(r, repeat=dim):
        ball.append([pm + k*step for k in x])
    center = random.choice(ball)

    # Calculates surrogate points and plots results.
    print("Calculating surrogate points...")
    surrogate_points = {}
    for point in negative_samples:
        surrogate = find_surrogate(point, center, d2_hat, oracle)
        surrogate_points[tuple(point)] = tuple(surrogate)
        plt.plot(surrogate[0], surrogate[1], marker='o', markersize=1, color="blue")
    for point in positive_samples:
        plt.plot(point[0], point[1], marker='o', markersize=1, color="red")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.savefig("surr.png")
    plt.clf()
    print("Done calculating surrogate points.\n")

    print("Calculating hyperplanes...")
    hyperplanes = []
    while True:
        flag = True
        while len(negative_samples) > 0:
            point = negative_samples[0]
            surrogate = surrogate_points[tuple(point)]

            # "Blames" each negative sample on a certain hyperplane, using
            # the surrogate points we calculated before.
            same_side = []
            other_samples = []
            for neg in negative_samples:
                surrogate_neg = surrogate_points[tuple(neg)]
                mid = midpoint(surrogate, surrogate_neg)
                if oracle.label_point(mid):
                    other_samples.append(neg)
                else:
                    same_side.append(neg)

            # Uses LP to find a halfspace containing same_side,
            # but not containing positive_samples.
            sol, success = find_separating_hyperplane(same_side,
                                                      positive_samples)

            # If the LP fails or we already found the maximum number of
            # hyperplanes, d2_hat must not be low enough.
            if not success or len(hyperplanes) == num_hyperplanes:
                d2_hat /= 2
                flag = False
                break

            hyperplanes.append([sol[0:2], sol[2]])
            negative_samples = other_samples
        if flag:
            break
    print("Done calculating hyperplanes.\n")

    # Plots 2D results.
    if dim == 2:
        plot2d(hyperplanes, dataset)

    return hyperplanes


# Tests the algorithm
if __name__ == "__main__":
    dim = 2
    num_hyperplanes = 5
    eps = 0.1
    delta = 1e-2
    dist = "uniform"
    oracle = ExampleOracle()

    dataset, labels = generate_dataset(dim, num_hyperplanes, eps, delta,
                                       dist, oracle, dataset_size=1000)

    polly(dim, num_hyperplanes, dataset, labels, oracle)
