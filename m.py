from numpy import array, eye, outer, sqrt, dot
from numpy.linalg import eig, inv

A = 2*array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]])

# vector than spans Ker(A)
k = array([1, 1, 1])/sqrt(3)

# projection onto R(A)
P = eye(3) - outer(k, k)

# idea is that it should be possible to decompose A as A = S*P or A = P*S
S=A/P

# indeed
print "A", A, "\n"
print "S", S, "\n"
print "S*P", S*P, "\n"
print "P*S", P*S, "\n"

# A has lambda=0 cooresponding to vector in nullspace
# at the mooment I can't say more about the two other eigenvalues
# but their eigenvectors are in R(A)
# A is not invertible
A_eign, A_eigv = eig(A)
for i in range(3):
  lmbda = A_eign[i]
  eigv = A_eigv[:, i]
  if abs(lmbda) < 1E-15: # get it to [1, 1, 1] 
    print lmbda, eigv*eigv[0] 
  else: # do the R(A) test
    print lmbda, eigv, "Is in R(A)?", abs(dot(eigv, k)) < 1E-15
print

# let's see about eigenvalues
# there should be lambda = 0 for eigenvector in Ker(A)
# there should be lambda = 1 for eigenvector already in R(A),
# since dim(R(A)) = 2, there should two such eigenvectors
# P is not invertible
P_eign, P_eigv = eig(P)
for i in range(3):
  lmbda = P_eign[i]
  eigv = P_eigv[:, i]
  if abs(lmbda) < 1E-15: # get it to [1, 1, 1] 
    print lmbda, eigv*eigv[0] 
  else: # do the R(A) test
    print lmbda, eigv, "Is in R(A)?", abs(dot(eigv, k)) < 1E-15
print

# S is invertible
S_eign, S_eigv = eig(S)
for i in range(3):
  lmbda = S_eign[i]
  eigv = S_eigv[:, i]
  if abs(lmbda) < 1E-15: # get it to [1, 1, 1] 
    print lmbda, eigv*eigv[0] 
  else: # do the R(A) test
    print lmbda, eigv, "Is in R(A)?", abs(dot(eigv, k)) < 1E-15
print

b = array([-0.25, 0.5, -0.25])
x = dot(P, dot(inv(S), b))

print dot(A, x)

# A*x = b
# S*P*x = b
# P*x = S-1*b, now the thing is that x is chosen already in R(A) so that P*x = x
# the x = S-1*b
# x^{n+1} = S*Px^{n} - b // *inv(A)    and drive r to 0
# y - inv(A)*b = y - x if r = 0 then y = x
