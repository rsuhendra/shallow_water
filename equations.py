import numpy as np
from scipy import sparse

from field import Field, FieldSystem
from timesteppers import CrankNicolson, PredictorCorrector
from spatial import FiniteDifferenceUniformGrid, Left, Right

class SW: # no bottom topography, viscocity, boundary conditions

    def __init__(self, X, spatial_order, g,f,b): # g=gravity, f=coriolis, b=drag
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

        self.F_ops = [-(h * dudx + u * dhdx + h * dvdy + v * dhdy),
                      -(u * dudx + v * dudy + g * dhdx) + f * v - b * u,
                      -(u * dvdx + v * dvdy + g * dhdy) - f * u - b * v]
        self.BCs=[]

class SW2:  # no viscocity, boundary conditions,

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

        self.F_ops = [-h * dudx - u * dhdx - h * dvdy - v * dhdy -H*dudx -H*dvdy - u*dHdx -v*dHdy,
                      -u * dudx - v * dudy - g * dhdx + f*v -b*u,
                      -u * dvdx - v * dvdy - g * dhdy - f*u -b*v]

        self.BCs=[]

class SWsqBC: # SW with boundary conditions on a square tub

    def __init__(self, X, spatial_order, g,f,b):
        u = X.field_list[0]
        v = X.field_list[1]
        h = X.field_list[2]
        self.domain = u.domain
        self.X = X

        dhdx = FiniteDifferenceUniformGrid(1, spatial_order, h, axis=0)
        dhdy = FiniteDifferenceUniformGrid(1, spatial_order, h, axis=1)
        dudx = FiniteDifferenceUniformGrid(1, spatial_order, u, axis=0)
        dudy =  FiniteDifferenceUniformGrid(1, spatial_order, u, axis=1)
        dvdx = FiniteDifferenceUniformGrid(1, spatial_order, v, axis=0)
        dvdy = FiniteDifferenceUniformGrid(1, spatial_order, v, axis=1)

        self.F_ops = [-h * dudx - u * dhdx - h * dvdy - v * dhdy,
                      -u * dudx - v * dudy - g * dhdx + f * v - b * u,
                      -u * dvdx - v * dvdy - g * dhdy - f * u - b * v]

        self.BCs = [Left(0, spatial_order, u, 0, axis=0), Right(0, spatial_order, u, 0, axis=0),
                    Left(0, spatial_order, v, 0, axis=1), Right(0, spatial_order, v, 0, axis=1)]

class SW2sqBC: # SW2 with boundary conditions on a square tub
    # implement bc for H, right now only works for periodic H, eg sinx*siny

    def __init__(self, X, spatial_order, g, f, b,H):  # g=gravity, f=coriolis, b=drag
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

        self.F_ops = [-h * dudx - u * dhdx - h * dvdy - v * dhdy - H * dudx - H * dvdy - u * dHdx - v * dHdy,
                      -u * dudx - v * dudy - g * dhdx + f * v - b * u,
                      -u * dvdx - v * dvdy - g * dhdy - f * u - b * v]

        self.BCs = [Left(0, spatial_order, u, 0, axis=0), Right(0, spatial_order, u, 0, axis=0),
                    Left(0, spatial_order, v, 0, axis=1), Right(0, spatial_order, v, 0, axis=1)]