"""
Codes for getting normal vector of BoxMesh(*mins, *max).
"""

from dolfin import *
from numpy import zeros

class Normal(Expression):
  def __init__(self, mesh):
    Expression.__init__(self)
    X = mesh.coordinates()
    xi = X[:, 0].min()
    xa = X[:, 0].max()
    yi = X[:, 1].min()
    ya = X[:, 1].max()
    zi = X[:, 2].min()
    za = X[:, 2].max()

    self.limits = [[xi, xa], [yi, ya], [zi, za]]
  def value_shape(self):
    return (3, )

  def eval(self, values, x):
    limits = self.limits
    values[:] = 0
    
    for i in range(3):
      if near(x[i], limits[i][0]) or near(x[i], limits[i][1]):
        if near(x[i], limits[i][0]):
          values[i] = -1
        else:
          values[i] = 1
        break

    mag = sqrt(sum(v**2 for v in values)) + DOLFIN_EPS
    values /= mag


mesh = BoxMesh(0, 0, 0, 1, 1, 1, 10, 10, 10)
V = VectorFunctionSpace(mesh, "CG", 1)

n = Normal(mesh)
n = project(n, V)
plot(n, interactive=True)

print assemble(inner(n, n)*ds)



