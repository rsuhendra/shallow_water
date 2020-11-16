import numpy as np
from scipy import sparse

from field import Field, FieldSystem
from timesteppers import CrankNicolson, PredictorCorrector
from spatial import FiniteDifferenceUniformGrid, Left, Right

class linearSW:  # no viscocity, boundary conditions,

    def __init__(self, X, spatial_order, g, f,H): # g=gravity, f=coriolis, b=drag
        u = X.field_list[0]
        v = X.field_list[1]
        h = X.field_list[2]

        self.domain = u.domain
        self.X = X

        dhdx = FiniteDifferenceUniformGrid(1, spatial_order, h, axis=0)
        dhdy = FiniteDifferenceUniformGrid(1, spatial_order, h, axis=1)
        dudx = FiniteDifferenceUniformGrid(1, spatial_order, u, axis=0)
        dvdy = FiniteDifferenceUniformGrid(1, spatial_order, v, axis=1)

        self.F_ops = [-H*dudx -H*dvdy,
                      - g * dhdx + f*v ,
                      - g * dhdy - f*u]
        self.BCs=[]


class linearSW1D:  # no viscocity, boundary conditions, second dimension

    def __init__(self, X, spatial_order, g, f): # g=gravity, f=coriolis, b=drag
        u = X.field_list[0]
        v = X.field_list[1]
        h = X.field_list[2]
        # v = X.field_list[2]

        self.domain = u.domain
        self.X = X

        dhdx = FiniteDifferenceUniformGrid(1, spatial_order, h, axis=0)
        # dhdy = FiniteDifferenceUniformGrid(1, spatial_order, h, axis=1)
        dudx = FiniteDifferenceUniformGrid(1, spatial_order, u, axis=0)
        # dvdy = FiniteDifferenceUniformGrid(1, spatial_order, v, axis=1)

        self.F_ops = [-H*dudx,
                      - g * dhdx]
                      # - g * dhdy - f*u]
        self.BCs=[]
