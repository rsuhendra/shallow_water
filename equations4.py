import numpy as np
from scipy import sparse


from field import Field, FieldSystem
from timesteppers import CrankNicolson, PredictorCorrector
from spatial import FiniteDifferenceUniformGrid, Left, Right


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