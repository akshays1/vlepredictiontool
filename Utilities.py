#!/usr/bin/env python
import scipy
import pylab
def sigdig_n(x, n):
    '''
    This is a method to return x correct to n>=1 significant digits.
    The value returned is a float. 
    '''
    if n < 1:
        raise ValueError("number of significant digits >= 1")
    format = '%.'+str(n-1)+'e'
    y = format%x
    return float(y)

import math
import scipy



def cubicsolver(parameterslist):
#for thermodynamics calculations only!
        try:
        
            b=parameterslist[0]
            c=parameterslist[1]
            d=parameterslist[2]
            root11=[0,0,0]
            p=(1.0/3)*(3*c-b*b)
            q=(1.0/27)*(27*d-9*b*c+2*b*b*b)
            R=(p/3.0)**3+(q/2.0)**2

            if R>=0:
                aa=-(q/2.0)+math.sqrt(R)
                bb=-(q/2.0)-math.sqrt(R)
                if aa>=0:
                    A=aa**(1.0/3)
                else:
                    A=-(-aa)**(1.0/3)
                if bb>=0:
                    B=bb**(1.0/3)
                else:
                    B=-(-bb)**(1.0/3)
                root11[0]=A+B-(b/3.0)
                root11[1]=-.5*(A+B)+complex(0,0.5*(3.0**0.5)*(A-B))-(b/3.0)
                root11[2]=-.5*(A+B)-complex(0,0.5*(3.0**0.5)*(A-B))-(b/3.0)
                if R>0:
    
                    end=[]
                    for a in root11:
                        if a.imag<0.0001:
                            end.append(a.real)
                    return end
                    
                else:
                    print'all roots real, atleast two equal'
                    return root11
            else:
                    ab=0.25*q*q
                    cd=-p*p*p/27.0
                    jj=scipy.sqrt(ab/cd)
                    phi=scipy.arccos(jj)
                    for k in range(0,3):
                            if q>0:
                                root11[k]=-2*scipy.sqrt(-p/3.0)*scipy.cos((phi/3.0)+(2*k*(scipy.pi)/3))-(b/3.0)
                            else:
                                root11[k]=2*scipy.sqrt(-p/3.0)*scipy.cos((phi/3.0)+(2*k*(scipy.pi)/3))-(b/3.0)
                    


                    return root11
        except  TypeError:
            print'ERROR: You have entered an incompatible argument.Please enter real arguments in list form.'
    


def evaluatepolynomial(x, listp):
    '''
    evaluatepolynomial(x, listp)
    
    Evaluate polynomial at x using coefficients in listp
        INPUTS
            x     =  Scalar.  Point at which to evaluate
            listp =  List of polynomial coefficients [a0, a1, ... , an]
        OUTPUT
            y     =  a0 + a1*x + ... + an*x**n
    '''
    y = 0.0    
    for i in range(len(listp)):
        p = float(listp[i])
        y += p*x**i
    return y
    
def fitpolynomial(listx, listy, order, test):
    '''
    fitpolynomial(listx, listy, order, test)
        INPUTS
            listx  =  a list of x coordinates of the curve to be fitted
            listy  =  a list of the corresponding y coordinates
            order  =  integer order of the polynomial.
                        p = a0 + a1*x + a2*x**2 + ... + an*x**n;   n = order
            test   =  boolean; either True or False.
                        if True, then the fitted curve is plotted else not.
        OUTPUTS
            listp  =  a list of polynomial coefficients [a0, a1, a2 ...]
    '''
    def funceval(listp, listx, listy):
        error = []
        for i in range(len(listx)):
            x = listx[i]
            y = listy[i]
            yrecalc = evaluatepolynomial(x, listp)
            error.append(y - yrecalc)
        return error
    listpguess = scipy.array((order+1)*[0.0])
    plsq = scipy.optimize.leastsq(funceval, listpguess, args=(listx, listy))
    listp = plsq[0]
    if test:
        xx = scipy.linspace(min(listx), max(listx), 1000)
        yy = evaluatepolynomial(xx, listp)
        pylab.plot(listx, listy, 'ro')
        pylab.plot(xx, yy, 'b')
        pylab.xtitle('junk')
        pylab.ytitle('more junk')
        pylab.title('The JUNK')
        pylab.show()
        pylab.clf()
    return listp
if __name__ == "__main__":
#    b, c, d = -6.0, 11.0, -6.0
#    yreal, ycplx = cubicsolver(b,c,d)
#    print yreal
#    print  ycplx
#    for y in yreal+ycplx:
#        print y, y**3 + b*y**2 + c*y + d
    xx = scipy.linspace(0.0, 20.0, 30)
    yy = xx/(1+xx)
    listp = fitpolynomial(xx, yy, 3, test=True)
    
