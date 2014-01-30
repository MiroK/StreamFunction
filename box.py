from dolfin import *
from numpy import zeros

u0 = ('2./sqrt(3.)*sin(2.*pi/3.)*sin(x[0])*cos(x[1])*cos(x[2])',
       '2./sqrt(3.)*sin(-2.*pi/3.)*cos(x[0])*sin(x[1])*cos(x[2])',
       '1.0')

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
    self.h = mesh.hmax()

    self.limits = [[xi, xa], [yi, ya], [zi, za]]
  def value_shape(self):
    return (3, )

  def eval(self, values, x):
    limits = self.limits

    values[:] = 0
    
    for i in range(3):
      if near(x[i], limits[i][0], h) or near(x[i], limits[i][1], h):
        if near(x[i], limits[i][0], h):
          values[i] = -1
        else:
          values[i] = 1
        break

    mag = sqrt(sum(v**2 for v in values)) + DOLFIN_EPS
    values /= mag
  
  def eval_cell(self, values, x, ufl_cell):
    limits = self.limits
    h = self.h

    values[:] = 0
    
    M = Cell(mesh, ufl_cell.index).midpoint()
    y = zeros(3)
    y[0] = M.x(); y[1] = M.y(); y[2] = M.z();
    
    for i in range(3):
      if near(y[i], limits[i][0], h) or near(y[i], limits[i][1], h):
        if near(y[i], limits[i][0], h):
          values[i] = -1
        else:
          values[i] = 1
        break

    mag = sqrt(sum(v**2 for v in values)) + DOLFIN_EPS
    values /= mag

mesh = BoxMesh(-pi, -pi, -pi, pi, pi, pi, 16, 16, 16)
V = VectorFunctionSpace(mesh, 'CG', 1)

n = Normal(mesh); n = project(n, V)

plot(inner(n, n), interactive=True)

#n = FacetNormal(mesh) #gmres jacobi
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
solver.parameters['maximum_iterations'] = 800
solver.parameters['relative_tolerance'] = 1e-8
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
