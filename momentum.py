import numpy as np
import copy
import array
from numpy.linalg import norm
import math

def myGradientDescentMomentum(fnon, jac, x0, tol, maxk, gam, *fnonargs):
    # fnon     - name of the nonlinear function f(x)
    # jac      - name of the Jacobian function J(x)
    # x0       - initial guess for the solution x0
    # tol      - stopping tolerance for Newton's iteration
    # maxk     - maximum number of Newton iterations before stopping
    # gam      - value for gamma
    # fnonargs - optional arguments that will be passed to the nonlinear function
    k = 0
    x = x0

    F = eval(fnon)(x,*fnonargs)

    print(' k    f(xk)')

    xArray = []
    lastDelta = 0
    delta = 0
    gamma = gam

    # Main gradient descent loop
    while (norm(F,2) > tol and k <= maxk):
        # Evaluate Jacobian matrix
        J = eval(jac)(x,2,fnon,F,*fnonargs)

        delta = delta*gamma + 2 * np.matmul(np.transpose(J), F) #d_k = γd^k−1 + 2J_T F,
        x = x - 0.01 * delta #xk+1 = xk − λdk

        F = eval(fnon)(x,*fnonargs)


        xArray.append(x)
        k += 1
        print('{0:2.0f}  {1:2.2e}'.format(k, norm(F,2)))

    if (k >= maxk):
        print('Not converged')
    else:
        print('Converged to ')
        print(x)
    # Return the iterates as a Numpy array
    return(xArray)

def example2(x):
    F = np.zeros((2,1), dtype=np.float64)
    F[0] = x[0]**2 + 2*(x[1]**2) + math.sin(2*x[0])
    F[1] = x[0]**2 + math.cos(x[0] + 5*x[1]) - 1.2
    return F
#F1/x = 2x + 2cos(2x) F1/y = 4y
#F2/x = 2x - sin(x + 5y) F2/y = -5sin(x + 5y)
def dExample2(x,n,fnon,F0,*fnonargs):
    J = np.zeros((2,2), dtype=np.float64)
    J[0,0] = 2*x[0] + 2*math.cos(2*x[0])
    J[0,1] = 4*x[1]
    J[1,0] = 2*x[0] - math.sin(x[0] + 5*x[1])
    J[1,1] = -5*math.sin(x[0] + 5*x[1])
    return J


#myGradientDescentLS("example2", "dExample2", np.array([[-2], [-1]]), 10**-2, 100, 10);
for i in (0.3, 0.5, 0.8):
  xk = myGradientDescentMomentum("example2", "dExample2", np.array([[-2],[-1]]), 1e-2, 50, i)