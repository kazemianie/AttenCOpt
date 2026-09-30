#!/usr/bin/python

# Copyright 2017, Gurobi Optimization, Inc.

# Gurobi MIP benchmark for the wind farm maintenance scheduling problem,
# used to compare AttenCOpt's solution time/quality against the exact MIP
# formulation (paper Section VI-A).

import math
from gurobipy import Model, GRB, tuplelist, quicksum


def solve_maintenance_scheduling_mip(points, threads=0, timeout=None, gap=None):
    """
    Solves the wind farm maintenance scheduling problem to optimality using the MIP formulation
    :param points: list of (t, x, y) coordinate
    :return:
    """

    n = len(points)

    # Dictionary of Euclidean distance between each pair of points


    dist = {}
    for i in range(n):
        for j in range(n):
            for t in range(n):
                nxt = 0 if (t == n-1) else t+1
                if i != j:
                    dist[t, i, j] = math.sqrt(sum((points[t][i][k] - points[nxt][j][k]) ** 2 for k in range(2)))

    m = Model()
    m.Params.outputFlag = False
    # Create variables

    vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')

    # for i in range(n):
    #   m.addConstr(sum(vars[i,j] for j in range(n)) == 2)

    for i in range(n): # for a node there should 2 connections over the entire time horizon
        m.addConstr(sum(vars[t, i, j] + vars[t, j, i] for t in range(n) for j in range(n) if i != j ) == 2)

    for t in range(n): # At one time step there should be only one connection
        m.addConstr(sum(vars[t, i, j] for i in range(n) for j in range(n)  if i != j ) == 1)

    # If there is a connection between i,j at time t then
    # 1. i should be connected to some node at time t-1
    # 2. j should be connected to some node at time t+1
    for i in range(n):
        for j in range(n):
            if i != j:
                for t in range(n):
                    nxt = 0 if (t == n - 1) else t + 1
                    prev = n-1 if (t == 0) else t-1

                    m.addConstr(vars[t, i, j] <= sum(vars[nxt, j, k] for k in range(n) if k != j))
                    m.addConstr(vars[t, i, j] <= sum(vars[prev, k, i] for k in range(n) if k != i))

    # Optimize model

    m._vars = vars
    m.Params.lazyConstraints = 1
    m.Params.threads = threads
    if timeout:
        m.Params.timeLimit = timeout
    if gap:
        m.Params.mipGap = gap * 0.01  # Percentage
    m.optimize()

    vals = m.getAttr('x', vars)
    selected = tuplelist((t, i, j) for t, i, j in vals.keys() if vals[t, i, j] > 0.5)

    # sort the selected nodes in time
    tour = [i[1] for i in sorted(selected, key=lambda tup: tup[0])]

    assert len(tour) == n

    return m.objVal, tour
