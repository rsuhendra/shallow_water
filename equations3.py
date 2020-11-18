# Author: Carter Koehler and Richard Suhendra
#
# This file contains various forms of the 1D Shallow Water Equations with periodic boundary conditions
#
# Including:
# linearSW1D(X,spatial_order,g,H)
# SWFull1D(X,spatial_order,g,f,b,nu,H)
#
# Input variables are described here:
# X = field_list([u,h])
# spatial_order = desired spatial_accuracy of derivatives
# g = gravitational acceleration
# f = coriolis effect coefficient
# b = drag coefficient
# H = bottom topography

import numpy as np
from scipy import sparse

from field import Field, FieldSystem
from timesteppers import CrankNicolson, PredictorCorrector
from spatial import FiniteDifferenceUniformGrid, Left, Right


class linearSW1D:  # no viscocity, boundary conditions, second dimension

    def __init__(self, X, spatial_order, g,H): # g=gravity, f=coriolis, b=drag
        u = X.field_list[0]
        h = X.field_list[1]

        self.domain = u.domain
        self.X = X

        dhdx = FiniteDifferenceUniformGrid(1, spatial_order, h, axis=0)
        dudx = FiniteDifferenceUniformGrid(1, spatial_order, u, axis=0)
        dHdx = FiniteDifferenceUniformGrid(1, spatial_order, H, axis=0)

        self.F_ops = [-u*dudx- g * dhdx,
                      -u*dhdx - u*dHdx -h*dudx -H*dudx]

        self.BCs=[]

class SWFull1D:

    def __init__(self, X, spatial_order, g,f,b,H):
        self.t = 0
        self.iter = 0
        u = X.field_list[0]
        h = X.field_list[1]
        self.domain = u.domain
        self.X = X

        dhdx = FiniteDifferenceUniformGrid(1, spatial_order, h, axis=0)
        dudx = FiniteDifferenceUniformGrid(1, spatial_order, u, axis=0)
        dHdx = FiniteDifferenceUniformGrid(1, spatial_order, H, axis=0)
        dhdx2 = FiniteDifferenceUniformGrid(2, spatial_order, h, axis=0)

        lhs_op = SWLHS1D(X, H, dhdx, dHdx, dhdx2, g)
        nonlinear = SWF1D(X, dudx, dhdx, dHdx, f, g, b, H)

        self.ts_x = CrankNicolson(lhs_op)
        self.ts_f = PredictorCorrector(nonlinear)

    def step(self, dt):
        self.ts_f.step(dt / 2)
        self.ts_x.step(dt)
        self.ts_f.step(dt / 2)

        self.t += dt
        self.iter += 1

class SWLHS1D:
    def __init__(self, X, H, dhdx, dHdx, dhdx2, g):
        u = X.field_list[0]
        h = X.field_list[1]
        self.domain = u.domain
        self.X = X

        ut = Field(u.domain)
        ht = Field(h.domain)

        eq1 = ut + g * dhdx
        eq2 = ht #+ 1e-3 * dhdx2

        self.M = sparse.bmat([[eq1.field_coeff(ut, axis=0), eq1.field_coeff(ht, axis=0)],
                              [eq2.field_coeff(ut, axis=0), eq2.field_coeff(ht, axis=0)]])

        self.L = sparse.bmat([[eq1.field_coeff(u, axis=0), eq1.field_coeff(h, axis=0)],
                              [eq2.field_coeff(u, axis=0), eq2.field_coeff(h, axis=0)]])


class SWF1D:
    def __init__(self, X, dudx, dhdx, dHdx, f, g, b, H):
        u = X.field_list[0]
        h = X.field_list[1]
        self.domain = u.domain
        self.X = X

        self.F_ops = [-u * dudx,
                      -u * dhdx - u * dHdx - (h+H) * dudx]
        self.BCs = []