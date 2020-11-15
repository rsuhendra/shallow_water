import numpy as np
import matplotlib.pyplot as plt
import field
import spatial
import timesteppers
from equations2 import SWFull


resolution=200
grid_x = field.UniformPeriodicGrid(resolution, 20)
grid_y = field.UniformPeriodicGrid(resolution, 20)
domain = field.Domain((grid_x, grid_y))
x, y = domain.values()

IC = np.exp(-(x+(y-10)**2-14)**2/8)*np.exp(-((x-10)**2+(y-10)**2)/10)
BT = 0*x + 0.1

h = field.Field(domain)
u = field.Field(domain)
v = field.Field(domain)
H = field.Field(domain)
X = field.FieldSystem([u,v,h,H])
h.data[:] = IC
u.data[:] = 0*IC
v.data[:] = 0*IC
H.data[:] = BT

g = 9.81
nu = 0
f = 0
b = 0
alpha = 0.1

sw_problem = SWFull(X, 2, g, f, b, nu)
dt = alpha*grid_x.dx

while sw_problem.t < 1 - 1e-5:
    sw_problem.step(dt)

