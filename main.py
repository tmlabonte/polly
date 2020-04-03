import math
import random

import numpy as np
from scipy.optimize import linprog

from oracle import ExampleOracle

def midpoint(a, b):
    return (a + b) / 2

def generate_dataset(dim, num_hyperplanes, eps, delta, dist, oracle):
    term1 = 2 / eps * math.log(4 / delta, 2)
    term2 = 16 * (dim + 1) * num_hyperplanes * math.log(3 * num_hyperplanes, 2)
    term2 /= eps
    term2 *= math.log(13 / eps, 2)

    # term 2 almost always?
    num_samples = math.ceil(max(term1, term2))

    # too big for testing
    num_samples = 1000

    if dist == "uniform":
        dataset = np.random.random_sample((num_samples, dim))

    labels = np.array([oracle.label_point(p) for p in dataset])

    return dataset, labels

def find_surrogate(point, center, d2_hat, oracle):
    # binary search and membership queries to find surrogate point
    # and surrogate center with distance < d2_hat
    vec = center - point
    vec /= np.linalg.norm(vec)
    d2_hat_vec = d2_hat * vec
    surrogate = (point + center) / 2

    lo = point
    hi = center
    while oracle.label_point(surrogate) or not oracle.label_point(surrogate + d2_hat_vec):
        old_surrogate = surrogate
        if oracle.label_point(surrogate):
            surrogate = midpoint(surrogate, lo)
            hi = old_surrogate
        else:
            surrogate = midpoint(surrogate, hi)
            lo = old_surrogate

    return surrogate

def polly(dim, num_hyperplanes, dataset, labels, oracle):
    # Guess at max distance of surrogate points to concept boundary
    d2_hat = math.sqrt(dim) / (24 * len(dataset) * math.pow(num_hyperplanes, 2))
    d2_hat *= math.pow((1 / (math.pow(dim, 1.5) * len(dataset))), dim)

    # regular d2_hat is extremely low and can't really be handled
    if d2_hat < 1e-16:
        d2_hat = 1e-16

    positive_samples = dataset[labels]
    negative_samples = dataset[~labels]

    pm = positive_samples.mean()

    # d is min distance b/w polytope and samples
    # how to actually get this?
    d = 1e-3
    step = 2 * d / (len(dataset) * math.sqrt(dim))
    step /= 10

    # how to extend to >2 dim?
    r = range(-5, 5)

    # a ball fully included in P for choosing center
    ball = [(pm + a * step, pm + b * step) for b in r for a in r]

    center = random.choice(ball)

    surrogate_points = {}
    for i, point in enumerate(negative_samples):
        print(str(i+1) + "/" + str(len(negative_samples)))
        surrogate = find_surrogate(point, center, d2_hat, oracle)
        surrogate_points[point] = surrogate

    hyperplanes = []
    while negative_samples:
        point = negative_samples[0]
        surrogate = surrogate_points[point]

        same_side = [b in negative_samples if not oracle.label_point(midpoint(surrogate, surrogate_points[b]))]

        # Use LP to find a halfspace containing same_side but not positive_samples
        sol = linprog()[0]
        hyperplanes.append



if __name__ == "__main__":
    dim = 2
    num_hyperplanes = 4
    eps = 0.05
    delta = 1e-3
    dist = "uniform"
    oracle = ExampleOracle()

    dataset, labels = generate_dataset(dim, num_hyperplanes,
                                       eps, delta, dist, oracle)

    polly(dim, num_hyperplanes, dataset, labels, oracle)
