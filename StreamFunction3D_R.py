from dolfin import *

mesh = BoxMesh(-pi, -pi, -pi, pi, pi, pi, 16, 16, 16)

# Declare solution Functions and FunctionSpaces
V = VectorFunctionSpace(mesh, 'CG', 1)
R = VectorFunctionSpace(mesh, 'R', 0)
VR = V*R

u0 = ('2./sqrt(3.)*sin(2.*pi/3.)*sin(x[0])*cos(x[1])*cos(x[2])',
       '2./sqrt(3.)*sin(-2.*pi/3.)*cos(x[0])*sin(x[1])*cos(x[2])',
       '0.0', '0.0', '0.0', '0.0')
uc_ = interpolate(Expression(u0, element=VR.ufl_element()), VR)
u_, c_ = split(uc_)

q, d = TestFunctions(VR)
psi, c = TrialFunctions(VR)
n = FacetNormal(mesh)

a = inner(grad(q), grad(psi))*dx - inner(q, dot(n, grad(psi)))*ds + (dot(psi, d) + dot(q, c))*dx
L = inner(q, curl(u_))*dx - dot(q, cross(n, u_))*ds

# Compute solution
psic = Function(VR)
psi, c = split(psic)
solve(a == L, psic)

A = assemble(a)
print A.array().shape

plot(psi[0])
plot(psi[1])
plot(psi[2])
uu = project(curl(psi), V, bcs=[DirichletBC(V, u_, DomainBoundary())])
plot(u_, title='u exact')
plot(uu, title="u from psi")

plot(u_-uu)

R = VectorFunctionSpace(mesh, 'R', 0)
c = TestFunction(R)
print "Assemble:"
print assemble(dot(psi, c)*dx).array()
print "Mean value:"
pp = psic.split(True)[0].split(True)
for p in pp:
    print p.vector().array().mean()

interactive()
