import numpy as np
from scipy import sparse

from field import Field, FieldSystem
from timesteppers import CrankNicolson, PredictorCorrector,BackwardEuler
from spatial import FiniteDifferenceUniformGrid, Left, Right

class SWlinearBE:

    def __init__(self, X, spatial_order, g,H):
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
        dvdy = FiniteDifferenceUniformGrid(1, spatial_order, v, axis=1)


        dx = SWx(X, g, H, dudx, dhdx)
        dy = SWy(X, g,H,dvdy, dhdy)

        self.ts_x = BackwardEuler(dx, 0)
        self.ts_y = BackwardEuler(dy, 1)

    def step(self, dt):
        self.ts_x.step(dt / 2)
        self.ts_y.step(dt / 2)
        self.ts_y.step(dt / 2)
        self.ts_x.step(dt / 2)

        self.t += dt
        self.iter += 1

class SWx:
    def __init__(self, X,g,H,dudx,dhdx):
        u = X.field_list[0]
        v = X.field_list[1]
        h = X.field_list[2]
        self.domain = u.domain
        self.X = X

        ut = Field(u.domain)
        vt = Field(v.domain)
        ht = Field(h.domain)

        eq1 = ut + g*dhdx
        eq2 = vt
        eq3 = ht + H*dudx

        self.M = sparse.bmat([[eq1.field_coeff(ut, axis=0), eq1.field_coeff(vt, axis=0), eq1.field_coeff(ht, axis=0)],
                              [eq2.field_coeff(ut, axis=0), eq2.field_coeff(vt, axis=0), eq2.field_coeff(ht, axis=0)],
                              [eq3.field_coeff(ut, axis=0), eq3.field_coeff(vt, axis=0), eq3.field_coeff(ht, axis=0)]])

        self.L = sparse.bmat([[eq1.field_coeff(u, axis=0), eq1.field_coeff(v, axis=0), eq1.field_coeff(h, axis=0)],
                              [eq2.field_coeff(u, axis=0), eq2.field_coeff(v, axis=0), eq2.field_coeff(h, axis=0)],
                              [eq3.field_coeff(u, axis=0), eq3.field_coeff(v, axis=0), eq3.field_coeff(h, axis=0)]])

class SWy:
    def __init__(self, X, g,H,dvdy,dhdy):
        u = X.field_list[0]
        v = X.field_list[1]
        h = X.field_list[2]
        self.domain = u.domain
        self.X = X

        ut = Field(u.domain)
        vt = Field(v.domain)
        ht = Field(h.domain)

        eq1 = ut
        eq2 = vt +g*dhdy
        eq3 = ht +H*dvdy

        self.M = sparse.bmat([[eq1.field_coeff(ut, axis=1), eq1.field_coeff(vt, axis=1), eq1.field_coeff(ht, axis=1)],
                              [eq2.field_coeff(ut, axis=1), eq2.field_coeff(vt, axis=1), eq2.field_coeff(ht, axis=1)],
                              [eq3.field_coeff(ut, axis=1), eq3.field_coeff(vt, axis=1), eq3.field_coeff(ht, axis=1)]])

        self.L = sparse.bmat([[eq1.field_coeff(u, axis=1), eq1.field_coeff(v, axis=1), eq1.field_coeff(h, axis=1)],
                              [eq2.field_coeff(u, axis=1), eq2.field_coeff(v, axis=1), eq2.field_coeff(h, axis=1)],
                              [eq3.field_coeff(u, axis=1), eq3.field_coeff(v, axis=1), eq3.field_coeff(h, axis=1)]])