from dolfin import *

class SphereNormal(Expression):
  def eval(self, value, x):
    mag = sqrt(sum(y**2 for y in x))
    for i in range(len(x)):
      value[i] = x[i]/mag
  
  def value_shape(self):
    return (3, )

mesh = SphereMesh(Point(0., 0., 0.), 1, 0.3)
V = VectorFunctionSpace(mesh, "CG", 1)

u = project(SphereNormal(), V)

plot(mesh, interactive=True)
plot(u, interactive=True)


