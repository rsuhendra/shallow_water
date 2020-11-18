# Author: Carter Koehler and Richard Suhendra
#
# This file contains various forms of the 2D Shallow Water Equations with non-periodic boundary conditions
#
# Including:
# SWBC(X,spatial_order,g,f,b,H)
# linearSWBC(X,spatial_order,g,f,b,H)
# SWFullBC(X,spatial_order,g,f,b,nu,H)
#
# Input variables are described here:
# X = field_list([u,v,h])
# spatial_order = desired spatial_accuracy of derivatives
# g = gravitational acceleration
# f = coriolis effect coefficient
# b = drag coefficient
# nu = viscocity coefficient

import numpy as np
from scipy import sparse

from field import Field, FieldSystem
from timesteppers import CrankNicolson, PredictorCorrector
from spatial import FiniteDifferenceUniformGrid, Left, Right

class SWBC:  # no viscocity

    def __init__(self, X, spatial_order, g, f,b,H): # g=gravity, f=coriolis, b=drag
        u = X.field_list[0]
        v = X.field_list[1]
        h = X.field_list[2]
        self.domain = u.domain
        self.X = X

        dhdx = FiniteDifferenceUniformGrid(1, spatial_order, h, axis=0)
        dhdy = FiniteDifferenceUniformGrid(1, spatial_order, h, axis=1)
        dudx = FiniteDifferenceUniformGrid(1, spatial_order, u, axis=0)
        dudy = FiniteDifferenceUniformGrid(1, spatial_order, u, axis=1)
        dvdx = FiniteDifferenceUniformGrid(1, spatial_order, v, axis=0)
        dvdy = FiniteDifferenceUniformGrid(1, spatial_order, v, axis=1)
        dHdx = FiniteDifferenceUniformGrid(1, spatial_order, H, axis=0)
        dHdy = FiniteDifferenceUniformGrid(1, spatial_order, H, axis=1)

        self.F_ops = [-u * dudx - v * dudy - g * dhdx + f*v -b*u,
                      -u * dvdx - v * dvdy - g * dhdy - f*u -b*v,
                      -h * dudx - u * dhdx - h * dvdy - v * dhdy -H*dudx -H*dvdy - u*dHdx -v*dHdy]

        self.BCs = [Left(0, spatial_order, u, 0, axis=0), Right(0, spatial_order, u, 0, axis=0),
                    Left(0, spatial_order, v, 0, axis=1), Right(0, spatial_order, v, 0, axis=1)]

class linearSWBC:  # linear, no viscocity

    def __init__(self, X, spatial_order, g, f,b,H): # g=gravity, f=coriolis, b=drag
        u = X.field_list[0]
        v = X.field_list[1]
        h = X.field_list[2]

        self.domain = u.domain
        self.X = X

        dhdx = FiniteDifferenceUniformGrid(1, spatial_order, h, axis=0)
        dhdy = FiniteDifferenceUniformGrid(1, spatial_order, h, axis=1)
        dudx = FiniteDifferenceUniformGrid(1, spatial_order, u, axis=0)
        dvdy = FiniteDifferenceUniformGrid(1, spatial_order, v, axis=1)
        dHdx = FiniteDifferenceUniformGrid(1, spatial_order, H, axis=0)
        dHdy = FiniteDifferenceUniformGrid(1, spatial_order, H, axis=1)

        self.F_ops = [- g * dhdx + f*v -b*u,
                      - g * dhdy - f*u -b*v,
                      -H*dudx -H*dvdy -u*dHdx -v*dHdy]

        self.BCs = [Left(0, spatial_order, u, 0, axis=0), Right(0, spatial_order, u, 0, axis=0),
                    Left(0, spatial_order, v, 0, axis=1), Right(0, spatial_order, v, 0, axis=1)]

class SWFullBC: # full with bc

    def __init__(self, X, spatial_order, g,f,b,nu,H):
        self.t = 0
        self.iter = 0
        u = X.field_list[0]
        v = X.field_list[1]
        h = X.field_list[2]
        self.domain = u.domain
        self.X = X

        dhdx = FiniteDifferenceUniformGrid(1, spatial_order, h, axis=0)
        dhdy = FiniteDifferenceUniformGrid(1, spatial_order, h, axis=1)
        dudx = FiniteDifferenceUniformGrid(1, spatial_order, u, axis=0)
        dudy = FiniteDifferenceUniformGrid(1, spatial_order, u, axis=1)
        dvdx = FiniteDifferenceUniformGrid(1, spatial_order, v, axis=0)
        dvdy = FiniteDifferenceUniformGrid(1, spatial_order, v, axis=1)
        dHdx = FiniteDifferenceUniformGrid(1, spatial_order, H, axis=0)
        dHdy = FiniteDifferenceUniformGrid(1, spatial_order, H, axis=1)

        dudx2 = FiniteDifferenceUniformGrid(2, spatial_order, u, axis=0)
        dudy2 = FiniteDifferenceUniformGrid(2, spatial_order, u, axis=1)
        dvdx2 = FiniteDifferenceUniformGrid(2, spatial_order, v, axis=0)
        dvdy2 = FiniteDifferenceUniformGrid(2, spatial_order, v, axis=1)


        diffx = SWDiffxBC(X, spatial_order, nu, dudx2, dvdx2)
        diffy = SWDiffyBC(X, spatial_order, nu, dudy2, dvdy2)
        nonlinear = SWFBC(X, spatial_order, dudx, dudy, dvdx, dvdy, dhdx, dhdy, dHdx, dHdy,f,g,b,H)

        self.ts_x = CrankNicolson(diffx, 0)
        self.ts_y = CrankNicolson(diffy, 1)
        self.ts_f = PredictorCorrector(nonlinear)

    def step(self, dt):
        self.ts_f.step(dt / 2)
        self.ts_x.step(dt / 2)
        self.ts_y.step(dt / 2)
        self.ts_y.step(dt / 2)
        self.ts_x.step(dt / 2)
        self.ts_f.step(dt / 2)

        self.t += dt
        self.iter += 1

class SWDiffxBC:
    def __init__(self, X, spatial_order, nu, dudx2, dvdx2):
        u = X.field_list[0]
        v = X.field_list[1]
        h = X.field_list[2]
        self.domain = u.domain
        self.X = X
        N = len(u.data)
        print(N)
        ut = Field(u.domain)
        vt = Field(v.domain)
        ht = Field(h.domain)

        eq1 = ut - nu * dudx2
        eq2 = vt - nu * dvdx2
        eq3 = ht

        bc1 = Left(0, spatial_order, u, 0, axis=0)
        bc2 = Right(0, spatial_order, u, 0, axis=0)

        self.M = sparse.bmat([[eq1.field_coeff(ut, axis=0), eq1.field_coeff(vt, axis=0), eq1.field_coeff(ht, axis=0)],
                              [eq2.field_coeff(ut, axis=0), eq2.field_coeff(vt, axis=0), eq2.field_coeff(ht, axis=0)],
                              [eq3.field_coeff(ut, axis=0), eq3.field_coeff(vt, axis=0), eq3.field_coeff(ht, axis=0)]])


        self.L = sparse.bmat([[eq1.field_coeff(u, axis=0), eq1.field_coeff(v, axis=0), eq1.field_coeff(h, axis=0)],
                              [eq2.field_coeff(u, axis=0), eq2.field_coeff(v, axis=0), eq2.field_coeff(h, axis=0)],
                              [eq3.field_coeff(u, axis=0), eq3.field_coeff(v, axis=0), eq3.field_coeff(h, axis=0)]])

        self.M=self.M.tocsr()
        self.M[0, :] = np.concatenate((bc1.field_coeff(ut, axis=0), bc1.field_coeff(vt, axis=0), bc1.field_coeff(ht, axis=0)),axis=1)
        self.M[N-1,:] = np.concatenate((bc2.field_coeff(ut, axis=0), bc2.field_coeff(vt, axis=0), bc2.field_coeff(ht, axis=0)),axis=1)
        self.M.eliminate_zeros()

        self.L=self.L.tocsr()
        self.L[0, :] = np.concatenate((bc1.field_coeff(u, axis=0), bc1.field_coeff(v, axis=0), bc1.field_coeff(h, axis=0)),axis=1)
        self.L[N-1, :] = np.concatenate((bc2.field_coeff(u, axis=0), bc2.field_coeff(v, axis=0), bc2.field_coeff(h, axis=0)),axis=1)
        self.L.eliminate_zeros()

class SWDiffyBC:
    def __init__(self, X, spatial_order, nu, dudy2, dvdy2):
        u = X.field_list[0]
        v = X.field_list[1]
        h = X.field_list[2]
        self.domain = u.domain
        self.X = X
        N = len(u.data)

        ut = Field(u.domain)
        vt = Field(v.domain)
        ht = Field(h.domain)

        eq1 = ut - nu * dudy2
        eq2 = vt - nu * dvdy2
        eq3 = ht

        bc1 = Left(0, spatial_order, v, 0, axis=1)
        bc2 = Right(0, spatial_order, v, 0, axis=1)

        self.M = sparse.bmat([[eq1.field_coeff(ut, axis=1), eq1.field_coeff(vt, axis=1), eq1.field_coeff(ht, axis=1)],
                              [eq2.field_coeff(ut, axis=1), eq2.field_coeff(vt, axis=1), eq2.field_coeff(ht, axis=1)],
                              [eq3.field_coeff(ut, axis=1), eq3.field_coeff(vt, axis=1), eq3.field_coeff(ht, axis=1)]])

        self.L = sparse.bmat([[eq1.field_coeff(u, axis=1), eq1.field_coeff(v, axis=1), eq1.field_coeff(h, axis=1)],
                              [eq2.field_coeff(u, axis=1), eq2.field_coeff(v, axis=1), eq2.field_coeff(h, axis=1)],
                              [eq3.field_coeff(u, axis=1), eq3.field_coeff(v, axis=1), eq3.field_coeff(h, axis=1)]])

        self.M=self.M.tocsr()
        self.M[N, :] = np.concatenate((bc1.field_coeff(ut, axis=1), bc1.field_coeff(vt, axis=1), bc1.field_coeff(ht, axis=1)),axis=1)
        self.M[2*N-1, :] = np.concatenate((bc2.field_coeff(ut, axis=1), bc2.field_coeff(vt, axis=1), bc2.field_coeff(ht, axis=1)),axis=1)
        self.M.eliminate_zeros()

        self.L=self.L.tocsr()
        self.L[N, :] = np.concatenate((bc1.field_coeff(u, axis=1), bc1.field_coeff(v, axis=1), bc1.field_coeff(h, axis=1)),axis=1)
        self.L[2*N-1, :] = np.concatenate((bc2.field_coeff(u, axis=1), bc2.field_coeff(v, axis=1), bc2.field_coeff(h, axis=1)),axis=1)
        self.L.eliminate_zeros()

class SWFBC:
    def __init__(self, X,spatial_order, dudx, dudy, dvdx, dvdy, dhdx, dhdy, dHdx, dHdy,f,g,b,H):
        u = X.field_list[0]
        v = X.field_list[1]
        h = X.field_list[2]
        self.domain = u.domain
        self.X = X

        self.F_ops = [-u * dudx - v * dudy - g * dhdx + f * v - b * u,
                      -u * dvdx - v * dvdy - g * dhdy - f * u - b * v,
                      -h * dudx - u * dhdx - h * dvdy - v * dhdy - H * dudx - H * dvdy - u * dHdx - v * dHdy]

        self.BCs = [Left(0, spatial_order, u, 0, axis=0), Right(0, spatial_order, u, 0, axis=0),
                    Left(0, spatial_order, v, 0, axis=1), Right(0, spatial_order, v, 0, axis=1)]
