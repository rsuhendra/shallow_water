import numpy as np
from scipy import sparse

from field import Field, FieldSystem
from timesteppers import CrankNicolson, PredictorCorrector
from spatial import FiniteDifferenceUniformGrid, Left, Right

class SW:  # no viscocity, boundary conditions,

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

        self.BCs=[]

class linearSW:  # no viscocity, boundary conditions,

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

        self.F_ops = [- g * dhdx + f*v -b*u,
                      - g * dhdy - f*u -b*v,
                      -H*dudx -H*dvdy]
        self.BCs=[]

class linearSWBE: #no viscocity, boundary conditions

    def __init__(self, X, spatial_order, g,H): # g=gravity, f=coriolis, b=drag
        u = X.field_list[0]
        v = X.field_list[1]
        h = X.field_list[2]

        self.domain = u.domain
        self.X = X
        
        ut = Field(u.domain)
        vt = Field(v.domain)
        ht = Field(h.domain)
        
        dhdx = FiniteDifferenceUniformGrid(1, spatial_order, h, axis=0)
        dhdy = FiniteDifferenceUniformGrid(1, spatial_order, h, axis=1)
        dudx = FiniteDifferenceUniformGrid(1, spatial_order, u, axis=0)
        dvdy = FiniteDifferenceUniformGrid(1, spatial_order, v, axis=1)

        eq1 = ut + g*dhdx
        eq2 = vt + g*dhdy
        eq3 = ht + H*dudx + H*dvdy
        
        self.M = sparse.bmat([[eq1.field_coeff(ut), eq1.field_coeff(vt), eq1.field_coeff(ht)],
                              [eq2.field_coeff(ut), eq2.field_coeff(vt), eq2.field_coeff(ht)],
                              [eq3.field_coeff(ut), eq3.field_coeff(vt), eq3.field_coeff(ht)]])

        self.L = sparse.bmat([[eq1.field_coeff(u), eq1.field_coeff(v), eq1.field_coeff(h)],
                              [eq2.field_coeff(u), eq2.field_coeff(v), eq2.field_coeff(h)],
                              [eq3.field_coeff(u), eq3.field_coeff(v), eq3.field_coeff(h)]])

class SWFull:

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


        diffx = SWDiffx(X, nu, dudx2, dvdx2)
        diffy = SWDiffy(X, nu, dudy2, dvdy2)
        nonlinear = SWF(X, dudx, dudy, dvdx, dvdy, dhdx, dhdy, dHdx, dHdy,f,g,b,H)

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
        u = X.field_list[0]
        v = X.field_list[1]
        h = X.field_list[2]
        self.domain = u.domain
        self.X = X

        ut = Field(u.domain)
        vt = Field(v.domain)
        ht = Field(h.domain)

        eq1 = ut - nu * dudx2
        eq2 = vt - nu * dvdx2
        eq3 = ht

        self.M = sparse.bmat([[eq1.field_coeff(ut, axis=0), eq1.field_coeff(vt, axis=0), eq1.field_coeff(ht, axis=0)],
                              [eq2.field_coeff(ut, axis=0), eq2.field_coeff(vt, axis=0), eq2.field_coeff(ht, axis=0)],
                              [eq3.field_coeff(ut, axis=0), eq3.field_coeff(vt, axis=0), eq3.field_coeff(ht, axis=0)]])

        self.L = sparse.bmat([[eq1.field_coeff(u, axis=0), eq1.field_coeff(v, axis=0), eq1.field_coeff(h, axis=0)],
                              [eq2.field_coeff(u, axis=0), eq2.field_coeff(v, axis=0), eq2.field_coeff(h, axis=0)],
                              [eq3.field_coeff(u, axis=0), eq3.field_coeff(v, axis=0), eq3.field_coeff(h, axis=0)]])

class SWDiffy:
    def __init__(self, X, nu, dudy2, dvdy2):
        u = X.field_list[0]
        v = X.field_list[1]
        h = X.field_list[2]
        self.domain = u.domain
        self.X = X

        ut = Field(u.domain)
        vt = Field(v.domain)
        ht = Field(h.domain)

        eq1 = ut - nu * dudy2
        eq2 = vt - nu * dvdy2
        eq3 = ht

        self.M = sparse.bmat([[eq1.field_coeff(ut, axis=1), eq1.field_coeff(vt, axis=1), eq1.field_coeff(ht, axis=1)],
                              [eq2.field_coeff(ut, axis=1), eq2.field_coeff(vt, axis=1), eq2.field_coeff(ht, axis=1)],
                              [eq3.field_coeff(ut, axis=1), eq3.field_coeff(vt, axis=1), eq3.field_coeff(ht, axis=1)]])

        self.L = sparse.bmat([[eq1.field_coeff(u, axis=1), eq1.field_coeff(v, axis=1), eq1.field_coeff(h, axis=1)],
                              [eq2.field_coeff(u, axis=1), eq2.field_coeff(v, axis=1), eq2.field_coeff(h, axis=1)],
                              [eq3.field_coeff(u, axis=1), eq3.field_coeff(v, axis=1), eq3.field_coeff(h, axis=1)]])

class SWF:
    def __init__(self, X, dudx, dudy, dvdx, dvdy, dhdx, dhdy, dHdx, dHdy,f,g,b,H):
        u = X.field_list[0]
        v = X.field_list[1]
        h = X.field_list[2]
        self.domain = u.domain
        self.X = X

        self.F_ops = [-u * dudx - v * dudy - g * dhdx + f * v - b * u,
                      -u * dvdx - v * dvdy - g * dhdy - f * u - b * v,
                      -h * dudx - u * dhdx - h * dvdy - v * dhdy - H * dudx - H * dvdy - u * dHdx - v * dHdy]
        self.BCs = []
