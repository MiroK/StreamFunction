"""
Codes for getting normal vector and gradient of normal vector for ellipsoid
with axis a, b, c.
"""

from dolfin import *
from numpy import zeros

# ----------------------How to get normal vector?
# Let phi = x**2/a**2 + y**2/b**2 + z**2/c**1 - 1. Then phi = 0 is an implicit
# definition of the surface of ellipsoids. We let x` = x/a**2, y` = y/b**2,
# z` = z/c**2. Then n = grad(n) = 2[x`, y`, z`] and with normalization
# n = [x`, y`, z`]/sqrt(x`_i*x`_i)

ellipsoid_n="""
class EllipsoidNormal : public Expression
{
public:
  double a, b, c;

  EllipsoidNormal() : Expression(3){ }

  void eval(Array<double>& values, const Array<double>& x) const
  {
    double axis[3]; axis[0] = a; axis[1] = b; axis[2] = c;
  
    // denote the prime coordinates as y
    double y[3];
    for(int i = 0; i < 3; i++)
    {
      y[i] = 0;
      y[i] = x[i]/axis[i]/axis[i];
    }

    double mag = 0;
    for(int i = 0; i < 3; i++)
    {
      mag += y[i]*y[i];
    }
    mag = sqrt(mag) + DOLFIN_EPS;
    
    for(int i = 0; i < 3; i++)
      values[i] = y[i]/mag;
  }
};
"""

axis = zeros(3); axis[0] = 1; axis[1] = 1.25; axis[2] = 1;
full_mesh = EllipsoidMesh(Point(0., 0., 0.), axis, 0.1)
mesh = BoundaryMesh(full_mesh, "exterior")

V = VectorFunctionSpace(mesh, 'CG', 2)

n = Expression(ellipsoid_n)
n.a = axis[0];
n.b = axis[1];
n.c = axis[2];

n = project(n, V)
plot(n, interactive=True);

#------------------ How to get grad(n). 
# grad(n) = grad`(n).J where J is jacobian J_IJ = del x`_i/ del x_j
# and n = [x`, y`, z`]/sqrt(x`_i*x`_i), see sympy_help.py

ellipsoid_grad_n="""
class EllipsoidGradNormal : public Expression
{
public:
  double a, b, c;

  EllipsoidGradNormal() : Expression(3, 3){ }

  void eval(Array<double>& values, const Array<double>& x) const
  {
    double axis[3]; axis[0] = a; axis[1] = b; axis[2] = c;

    double y[3];
    for(int i = 0; i < 3; i++)
    {
      y[i] = x[i]/axis[i]/axis[i];
    }

    double mag = 0;
    for(int i = 0; i < 3; i++)
    {
      mag += y[i]*y[i];
    }
    mag = sqrt(mag);
    
    for(int i = 0; i < 3; i++)
    {
      for(int j = 0; j < 3; j++)
      {
        values[3*i + j] = 0;
        values[3*i + j] -= y[i]*y[j]/axis[j]/axis[j]/pow(mag, 3);
      }
      values[3*i + i] += 1/axis[i]/axis[i]/mag;
    }
  }
};
"""

S = TensorFunctionSpace(mesh, 'CG', 1)

grad_n = Expression(ellipsoid_grad_n)
grad_n.a = axis[0];
grad_n.b = axis[1];
grad_n.c = axis[2];

grad_n = project(grad_n, S)
for i in range(3):
  for j in range(3):
    plot(grad_n[i, j], interactive=True, title="[%d, %d]" % (i, j))

# this should give curvature ... really the only way I have to see if the
# gradient is computed correctly
plot(0.5*tr(grad_n), interactive=True) 


