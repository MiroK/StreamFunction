from dolfin import *
from numpy import zeros

u0 = ('2./sqrt(3.)*sin(2.*pi/3.)*sin(x[0])*cos(x[1])*cos(x[2])',
       '2./sqrt(3.)*sin(-2.*pi/3.)*cos(x[0])*sin(x[1])*cos(x[2])',
       '1.0')

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

#-----------------------------------------------------------------------

axis = zeros(3); axis[0] = pi; axis[1] = 1.25*pi; axis[2] = 1.5*pi;
mesh = EllipsoidMesh(Point(0., 0., 0.), axis, 0.25)

V = VectorFunctionSpace(mesh, 'CG', 1)
S = TensorFunctionSpace(mesh, 'CG', 1)

n = Expression(ellipsoid_n)
n.a = axis[0];
n.b = axis[1];
n.c = axis[2];

grad_n = Expression(ellipsoid_grad_n)
grad_n.a = axis[0];
grad_n.b = axis[1];
grad_n.c = axis[2];

n = project(n, V)
grad_n = project(grad_n, S)

set_log_level(PROGRESS)

#------------------------------------------------------------------------------
u_ = interpolate(Expression(u0), V)

q = TestFunction(V)
psi = TrialFunction(V)

L = inner(q, curl(u_))*dx - dot(q, cross(n, u_))*ds
a = inner(grad(q), grad(psi))*dx       # laplace term
e0 = - inner(q, dot(n, grad(psi)))*ds  # extra surface terms
e1 = - inner(dot(psi, grad_n), q)*ds
e2 = inner(dot(grad_n, psi), q)*ds
e3 = inner(cross(psi, curl(n)), q)*ds

a += e0 + e1 + e2 + e3

# solving first the laplace problem does not seeme to help convergence
psi = Function(V)
A = assemble(a)
b = assemble(L)

solver = KrylovSolver('bicgstab', 'ilu') # *amg, *hypre fails
                                         # ilu and sor work nicely
                                         # also bicgstab better then gmres

solver.parameters['error_on_nonconvergence'] = False
solver.parameters['monitor_convergence'] = True
solver.parameters['maximum_iterations'] = 300
solver.parameters['relative_tolerance'] = 1e-6
solver.set_operator(A)
dim = V.dim()
n_comp = V.num_sub_spaces() 
null_vectors = []
vv = Function(V)
for i in range(n_comp):
    null_vec_i = vv.vector().copy()
    V.sub(i).dofmap().set(null_vec_i, 1) 
    null_vec_i *= 1.0/null_vec_i.norm("l2")
    null_vectors.append(null_vec_i)

null_space = VectorSpaceBasis(null_vectors)
null_space.orthogonalize(b)

solver.set_nullspace(null_space)  # nullspaces are really important
solver.solve(psi.vector(), b)

# Test solution by projecting back to the velocity
uu = project(curl(psi), V, bcs=[DirichletBC(V, u_, DomainBoundary())])

#plot(u_, title='u exact', interactive=True)
plot(uu, title="u from psi", interactive=True)

# get the norm
u_.vector()[:] -= uu.vector()
print sqrt(assemble(inner(u_, u_)*dx))
