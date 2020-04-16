import math
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

from oracle import ExampleOracle

def graph(formula, x):
    y = formula(x)
    plt.plot(x, y)

def midpoint(a, b):
    try:
        return (a + b) / 2
    except:
        return [(i + j) / 2 for i, j in zip(a, b)]

def generate_dataset(dim, num_hyperplanes, eps, delta, dist, oracle):
    term1 = 2 / eps * math.log(4 / delta, 2)
    term2 = 16 * (dim + 1) * num_hyperplanes * math.log(3 * num_hyperplanes, 2)
    term2 /= eps
    term2 *= math.log(13 / eps, 2)

    # term 2 almost always?
    num_samples = math.ceil(max(term1, term2))

    # too big for testing
    num_samples = 500

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
    surrogate = midpoint(point, center)

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
    if d2_hat < 1e-8:
        d2_hat = 1e-8

    positive_samples = np.array(dataset[labels])
    negative_samples = np.array(dataset[~labels])

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
        surrogate_points[tuple(point)] = tuple(surrogate)
        plt.plot(surrogate[0], surrogate[1], marker='o', markersize=1, color="blue")
    for point in positive_samples:
        plt.plot(point[0], point[1], marker='o', markersize=1, color="red")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.savefig("surr.png")
    plt.clf()

    hyperplanes = []
    flag = True
    while True:
        while len(negative_samples) > 0:
            point = negative_samples[0]
            surrogate = surrogate_points[tuple(point)]

            same_side = np.array([neg for neg in negative_samples if not oracle.label_point(midpoint(surrogate, surrogate_points[tuple(neg)]))])
            other_samples = np.array([neg for neg in negative_samples if oracle.label_point(midpoint(surrogate, surrogate_points[tuple(neg)]))])

            # Use LP to find a halfspace containing same_side but not positive_samples
            c = [0] * (dim + 1)

            A = []
            for i, pos in enumerate(positive_samples):
                A.append(list(-pos) + [1])
            for i, sam in enumerate(same_side):
                A.append(list(sam) + [-1])

            b = [-1] * (len(positive_samples) + len(same_side))

            lp = linprog(c=c, A_ub=A, b_ub=b, bounds=[-np.inf, np.inf])

            if not lp["success"] or len(hyperplanes) == num_hyperplanes:
                d2_hat /= 2
                flag = False
                break

            sol = lp["x"]
            opt = lp["fun"]

            hyperplanes.append([sol[0:2], sol[2]])
            negative_samples = other_samples
        if flag:
            break

    if dim == 2:
        for h in hyperplanes:
            a1 = h[0][0]
            a2 = h[0][1]
            b = h[1]

            m = -a1 / a2
            b = b / a2
            print("y = {}x + {}".format(m, b))
            graph(lambda x: m*x+b, np.arange(0, 1, 0.01))
        for d in dataset:
            if oracle.label_point(d):
                plt.plot(d[0], d[1], marker='o', markersize=1, color="red")
            else:
                plt.plot(d[0], d[1], marker='o', markersize=1, color="blue")
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.0)
        plt.savefig("pred.png")



if __name__ == "__main__":
    dim = 2
    num_hyperplanes = 4
    eps = 0.5
    delta = 1e-3
    dist = "uniform"
    oracle = ExampleOracle()

    dataset, labels = generate_dataset(dim, num_hyperplanes,
                                       eps, delta, dist, oracle)

    polly(dim, num_hyperplanes, dataset, labels, oracle)
