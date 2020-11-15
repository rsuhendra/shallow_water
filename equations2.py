import numpy as np
from scipy import sparse


from field import Field, FieldSystem
from timesteppers import CrankNicolson, PredictorCorrector
from spatial import FiniteDifferenceUniformGrid, Left, Right


class SWFull:

    def __init__(self, X, spatial_order, g,f,b,nu):
        self.t = 0
        self.iter = 0
        h = X.field_list[0]
        u = X.field_list[1]
        v = X.field_list[2]
        H = X.field_list[3]
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


        diffx = SWDiffx(X, nu, dudx2, dvdx2)
        diffy = SWDiffy(X, nu, dudy2, dvdy2)
        nonlinear = SWF(X, dudx, dudy, dvdx, dvdy, dhdx, dhdy, dHdx, dHdy,f,g,b)

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


class SWDiffx:
    def __init__(self, X, nu, dudx2, dvdx2):
        h = X.field_list[0]
        u = X.field_list[1]
        v = X.field_list[2]
        H = X.field_list[3]
        self.domain = u.domain
        self.X = X

        ht = Field(h.domain)
        ut = Field(u.domain)
        vt = Field(v.domain)
        Ht = Field(H.domain)

        eq1 = ht
        eq2 = ut - nu * dudx2
        eq3 = vt - nu * dvdx2
        eq4 = Ht

        self.M = sparse.bmat([[eq1.field_coeff(ht, axis=0), eq1.field_coeff(ut, axis=0), eq1.field_coeff(vt, axis=0),
                               eq1.field_coeff(Ht, axis=0)],
                              [eq2.field_coeff(ht, axis=0), eq2.field_coeff(ut, axis=0), eq2.field_coeff(vt, axis=0),
                               eq2.field_coeff(Ht, axis=0)],
                              [eq3.field_coeff(ht, axis=0), eq3.field_coeff(ut, axis=0), eq3.field_coeff(vt, axis=0),
                               eq3.field_coeff(Ht, axis=0)],
                              [eq4.field_coeff(ht, axis=0), eq4.field_coeff(ut, axis=0), eq4.field_coeff(vt, axis=0),
                               eq4.field_coeff(Ht, axis=0)]])

        self.L = sparse.bmat([[eq1.field_coeff(h, axis=0), eq1.field_coeff(u, axis=0), eq1.field_coeff(v, axis=0),
                               eq1.field_coeff(H, axis=0)],
                              [eq2.field_coeff(h, axis=0), eq2.field_coeff(u, axis=0), eq2.field_coeff(v, axis=0),
                               eq2.field_coeff(H, axis=0)],
                              [eq3.field_coeff(h, axis=0), eq3.field_coeff(u, axis=0), eq3.field_coeff(v, axis=0),
                               eq3.field_coeff(H, axis=0)],
                              [eq4.field_coeff(h, axis=0), eq4.field_coeff(u, axis=0), eq4.field_coeff(v, axis=0),
                               eq4.field_coeff(H, axis=0)]])


class SWDiffy:
    def __init__(self, X, nu, dudy2, dvdy2):
        h = X.field_list[0]
        u = X.field_list[1]
        v = X.field_list[2]
        H = X.field_list[3]
        self.domain = u.domain
        self.X = X

        ht = Field(h.domain)
        ut = Field(u.domain)
        vt = Field(v.domain)
        Ht = Field(H.domain)

        eq1 = ht
        eq2 = ut - nu * dudy2
        eq3 = vt - nu * dvdy2
        eq4 = Ht

        self.M = sparse.bmat([[eq1.field_coeff(ht, axis=1), eq1.field_coeff(ut, axis=1), eq1.field_coeff(vt, axis=1),
                               eq1.field_coeff(Ht, axis=1)],
                              [eq2.field_coeff(ht, axis=1), eq2.field_coeff(ut, axis=1), eq2.field_coeff(vt, axis=1),
                               eq2.field_coeff(Ht, axis=1)],
                              [eq3.field_coeff(ht, axis=1), eq3.field_coeff(ut, axis=1), eq3.field_coeff(vt, axis=1),
                               eq3.field_coeff(Ht, axis=1)],
                              [eq4.field_coeff(ht, axis=1), eq4.field_coeff(ut, axis=1), eq4.field_coeff(vt, axis=1),
                               eq4.field_coeff(Ht, axis=1)]])

        self.L = sparse.bmat([[eq1.field_coeff(h, axis=1), eq1.field_coeff(u, axis=1), eq1.field_coeff(v, axis=1),
                               eq1.field_coeff(H, axis=1)],
                              [eq2.field_coeff(h, axis=1), eq2.field_coeff(u, axis=1), eq2.field_coeff(v, axis=1),
                               eq2.field_coeff(H, axis=1)],
                              [eq3.field_coeff(h, axis=1), eq3.field_coeff(u, axis=1), eq3.field_coeff(v, axis=1),
                               eq3.field_coeff(H, axis=1)],
                              [eq4.field_coeff(h, axis=1), eq4.field_coeff(u, axis=1), eq4.field_coeff(v, axis=1),
                               eq4.field_coeff(H, axis=1)]])


class SWF:
    def __init__(self, X, dudx, dudy, dvdx, dvdy, dhdx, dhdy, dHdx, dHdy,f,g,b):
        h = X.field_list[0]
        u = X.field_list[1]
        v = X.field_list[2]
        H = X.field_list[3]
        self.domain = u.domain
        self.X = X

        self.F_ops = [-h * dudx - u * dhdx - h * dvdy - v * dhdy -H*dudx -H*dvdy - u*dHdx -v*dHdy,
                      -u * dudx - v * dudy - g * dhdx + f*v -b*u,
                      -u * dvdx - v * dvdy - g * dhdy - f*u -b*v,
                      0*H]

        self.BCs = [0*u, 0*u, 0*u, 0*u]
