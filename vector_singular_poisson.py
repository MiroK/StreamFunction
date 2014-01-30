from dolfin import *
from numpy import array, zeros, arange
from numpy import dot as npdot
from numpy.linalg import eig, qr
from numpy import set_printoptions

set_printoptions(linewidth=100)

#mesh = BoxMesh(-pi, -pi, -pi, pi, pi, pi, 6, 6, 6)
mesh = UnitCubeMesh(5, 5, 5)
V = VectorFunctionSpace(mesh, "CG", 1)

u0 = ('2./sqrt(3.)*sin(2.*pi/3.)*sin(x[0])*cos(x[1])*cos(x[2])',
       '2./sqrt(3.)*sin(-2.*pi/3.)*cos(x[0])*sin(x[1])*cos(x[2])',
       '0.0')
u_ = interpolate(Expression(u0), V)

u = TrialFunction(V)
v = TestFunction(V)

n = FacetNormal(mesh)
a = inner(grad(u), grad(v))*dx - inner(v, dot(n, grad(u)))*ds 
L = inner(v, curl(u_))*dx - dot(v, cross(n, u_))*ds

A = assemble(a)
b = assemble(L)

solver = KrylovSolver('gmres', 'jacobi')
#solver.parameters['monitor_convergence'] = True
solver.parameters['maximum_iterations'] = 1000
solver.parameters['absolute_tolerance'] = 1e-8
solver.parameters['relative_tolerance'] = 1e-8
solver.set_operator(A)

u_QR = Function(V)
u_ONES = Function(V)
n = V.dim()

# QR
eign, eigv = eig(A.array())

null_space_dim = 0
basis_vectors = []
for i in range(n):
  if abs(eign[i].real) < 10*DOLFIN_EPS:
    null_space_dim += 1
    basis_vectors.append(eigv[:, i])
print "Nullspace has dim", null_space_dim, "\n"

# orthogonalize the eigenvectors
basis_vectors, R = qr(array(basis_vectors).T)
basis_vectors = basis_vectors.real

# test the vectors and build the nullspace for Krylov
print "Basis test"
null_vectors = []
for i in range(null_space_dim):
  print "Basis vector %d of nullspace ?" % i,
  print all(abs(npdot(A.array(), basis_vectors[:, i])) < 100*DOLFIN_EPS)

  fun = Function(V)
  fun_v = fun.vector()
  fun_v[:] = array(basis_vectors[:, i].tolist())
  plot(fun, interactive=True, title="QR basis")

  null_vectors.append(fun.vector())

null_space = VectorSpaceBasis(null_vectors)
null_space.orthogonalize(b)

solver.set_nullspace(null_space)
solver.solve(u_QR.vector(), b)

# ONES
n_comp = V.num_sub_spaces()
null_vectors = []
for i in range(n_comp):
  null_vec_i = Vector(n)
  V.sub(i).dofmap().set(null_vec_i, 1) 
  null_vec_i *= 1.0/null_vec_i.norm("l2")
  null_vectors.append(null_vec_i)
  print all(abs(npdot(A.array(), null_vec_i.array())) < 100*DOLFIN_EPS)

null_space = VectorSpaceBasis(null_vectors)
null_space.orthogonalize(b)

solver.set_nullspace(null_space)
solver.solve(u_ONES.vector(), b)

plot(curl(u_QR), interactive=True, title="QR")
plot(curl(u_ONES), interactive=True, title="ONES")
