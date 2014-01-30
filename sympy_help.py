from sympy import *
init_printing(use_unicode=True)

a, b, c = symbols("a b c")
x, y, z = symbols("x y z")
u, v, w = symbols("u v w")

phi = x**2/a**2 + y**2/b**2 + z**2/c**2

u = x/a**2
v = y/b**2
w = z/c**2

# get the normal vector
phi_x = diff(phi, x)
phi_y = diff(phi, y)
phi_z = diff(phi, z)

phi_x = phi_x/sqrt(phi_x**2 + phi_y**2 + phi_z**2)
phi_y = phi_y/sqrt(phi_x**2 + phi_y**2 + phi_z**2)
phi_z = phi_z/sqrt(phi_x**2 + phi_y**2 + phi_z**2)

n = [phi_x, phi_y, phi_z]
xyz = [x, y, z]
uvw = [u, v, w]

# get jacobian
J = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
for i in range(3):
  for j in range(3):
    J[i][j] = simplify(diff(uvw[i], xyz[j]))

#print n
#print J

u, v, w = symbols("u v w")
uvw = [u, v, w]

mag = sqrt(u**2 + v**2 + w**2)
n_ = [u/mag, v/mag, w/mag]


grad_ = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
for i in range(3):
  for j in range(3):
    grad_[i][j] = diff(n_[i], uvw[j])

#print grad_

# result
grad = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
for i in range(3):
  for j in range(3):
    for k in range(3):
      grad[i][j] += grad_[i][k]*J[k][j]

print grad
