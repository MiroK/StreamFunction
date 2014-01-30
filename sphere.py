from dolfin import *

u0 = ('2./sqrt(3.)*sin(2.*pi/3.)*sin(x[0])*cos(x[1])*cos(x[2])',
       '2./sqrt(3.)*sin(-2.*pi/3.)*cos(x[0])*sin(x[1])*cos(x[2])',
       '1.0')

sphere_code="""
class SphereNormal : public Expression
{
public:
  SphereNormal() : Expression(3){ }

  void eval(Array<double>& values, const Array<double>& x) const
  {
    double mag = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
    for(int i = 0; i < 3; i++)
      values[i] = x[i]/mag;
  }
};
"""

mesh = SphereMesh(Point(0., 0., 0.), pi, 0.25)
V = VectorFunctionSpace(mesh, 'CG', 1)

n = Expression(sphere_code);n = project(n, V)

#------------------------------------------------------------------------------
u_ = interpolate(Expression(u0), V)

q = TestFunction(V)
psi = TrialFunction(V)

L = inner(q, curl(u_))*dx - dot(q, cross(n, u_))*ds
a = inner(grad(q), grad(psi))*dx       # laplace term
e0 = - inner(q, dot(n, grad(psi)))*ds  # extra surface terms
e1 = - inner(dot(psi, grad(n)), q)*ds
e2 = inner(dot(grad(n), psi), q)*ds
e3 = inner(cross(psi, curl(n)), q)*ds

a += e0 + e1 + e2 + e3

# solving first the laplace problem does not seeme to help convergence
psi = Function(V)
A = assemble(a)
b = assemble(L)

solver = KrylovSolver('bicgstab', 'ilu') # *amg, *hypre fails
                                         # ilu and sor work nicely
                                         # also bicgstab better then gmres

solver.parameters['nonzero_initial_guess'] = True
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
#plot(uu, title="u from psi", interactive=True)

# get the norm
u_.vector()[:] -= uu.vector()
print sqrt(assemble(inner(u_, u_)*dx))
