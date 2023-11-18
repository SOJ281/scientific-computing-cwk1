import numpy as np
import math

def dekker(fnon, x0, tol, maxk, *fnonargs):
    # fnon     - name of the nonlinear function f(x)
    # dfdx     - name of the derivative function df(x)
    # x0       - initial guess for the solution x0
    # tol      - stopping tolerance for Newton's iteration
    # maxk     - maximum number of Newton iterations before stopping
    # fnonargs - optional arguments that will be passed to the nonlinear function (useful for additional function parameters)
    #Makes sure x0 is valid
    if eval(fnon)(x0[0],*fnonargs) * eval(fnon)(x0[1],*fnonargs) >= 0:
      print("Root not bracketed")
      return
    k = 1
    x = x0
    f = eval(fnon)(x[0],*fnonargs)

    #[a,b] array
    iter = []
    iter.append([0, x0[0]]) #b-1
    iter.append(x) #b0
    #just the value of b
    bestChoices = []
    bestChoices.append(x0[0])

    print(' k  xk          f(xk)')
    # Main Newton loop
    while (abs(f) > tol and k < maxk+1):
        #Step 2a:Midpoint and secant
        s = (iter[k][1] - iter[k-1][1]) / (eval(fnon)(iter[k][1],*fnonargs) - eval(fnon)(iter[k-1][1],*fnonargs))
        s = s * eval(fnon)(iter[k][1],*fnonargs)
        s = iter[k][1] - s
        m = (iter[k][0] + iter[k][1])/2

        #Step 2b: If Secant is reliable accept as the new b_k+1
        if (m <= s and s <= iter[k][1]):
        #if (m * iter[k][1] <= 0):
          x = [0, s]
        else:
          x = [0, m]

        #step 3: check which satisfies bracket condition
        if (eval(fnon)(iter[k][0],*fnonargs) * eval(fnon)(x[1],*fnonargs) <= 0):
          x = [iter[k][0], x[1]]
        elif (eval(fnon)(x[1],*fnonargs) * eval(fnon)(iter[k][1],*fnonargs) <= 0):
          x = [x[1], iter[k][1]]

        #Compare a and b, swap if a is better
        if abs(eval(fnon)(x[0],*fnonargs)) < abs(eval(fnon)(x[1],*fnonargs)):
          temp = x[1]
          x[1] = x[0]
          x[0] = temp

        iter.append(x)
        f = eval(fnon)(x[1],*fnonargs)
        bestChoices.append(x[1])
        k += 1
        print('{0:2.0f}  {1:2.8f}  {2:2.2e}'.format(k-1, x[1], abs(f)))

    if (k == maxk):
        print('Not converged')
    else:
        print('Converged')
    return bestChoices


def function(x):
    # Define function f(x) = x^4 - 4 x^3 + 6 x^2 - 3 d
    #return (-math.cos(x) + x**3 + 2*x**2 + 1)
    return (-np.cos(x) + x**3 + 2*x**2 + 1)

def dfunction(x):
    # Define function df(x) = 4 x^3 - 12 x^2 + 12 x
    return (math.sin(x) + 3*x**2 + 4*x)

xk = dekker('function', [-3,1], 10**-6, 40)