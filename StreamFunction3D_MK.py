from dolfin import *
from numpy import zeros, where, any
from numpy import dot as npdot

mesh = BoxMesh(-pi, -pi, -pi, pi, pi, pi, 16, 16, 16)

# Declare solution Functions and FunctionSpaces
V = VectorFunctionSpace(mesh, 'CG', 2)

# Start from previous solution if restart_folder is given
u0 = ('2./sqrt(3.)*sin(2.*pi/3.)*sin(x[0])*cos(x[1])*cos(x[2])',
       '2./sqrt(3.)*sin(-2.*pi/3.)*cos(x[0])*sin(x[1])*cos(x[2])',
       '0.0')
u_ = interpolate(Expression(u0), V)

q = TestFunction(V)
psi = TrialFunction(V)
n = FacetNormal(mesh)

a = inner(grad(q), grad(psi))*dx #- inner(q, dot(n, grad(psi)))*ds
L = inner(q, curl(u_))*dx - dot(q, cross(n, u_))*ds

#p = inner(grad(q), grad(psi))*dx
#P = assemble(p)

# Compute solution
psi = Function(V)
A = assemble(a)
b = assemble(L)

solver = KrylovSolver('gmres', 'petsc_amg')
solver.parameters['monitor_convergence'] = True
solver.parameters['maximum_iterations'] = 1000
solver.parameters['absolute_tolerance'] = 1e-8
solver.parameters['relative_tolerance'] = 1e-8
solver.set_operator(A)

dim = V.dim()
n_comp = V.num_sub_spaces() 
null_vectors = []
for i in range(n_comp):
    null_vec_i = Vector(dim)
    V.sub(i).dofmap().set(null_vec_i, 1) 
    null_vec_i *= 1.0/null_vec_i.norm("l2")
    null_vectors.append(null_vec_i)

null_space = VectorSpaceBasis(null_vectors)
solver.set_nullspace(null_space)
null_space.orthogonalize(b)

solver.solve(psi.vector(), b)
#plot(psi[0])
#plot(psi[1])
#plot(psi[2])

# Test solution by projecting back to the velocity
uu = project(curl(psi), V, bcs=[DirichletBC(V, u_, DomainBoundary())])
#uu = project(curl(psi), V)
#plot(u_, title='u exact')
plot(uu, title="u from psi")

#R = VectorFunctionSpace(mesh, 'R', 0)
#c = TestFunction(R)
#print "Assemble:"
#print assemble(dot(psi, c)*dx).array()
#print "Mean value:"
#pp = psi.split(True)
#for p in pp:
#    print p.vector().array().mean()
#    
interactive()
