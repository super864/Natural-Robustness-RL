import numpy as np
from fractions import Fraction
import time
import math 

import copy 


def Bisection(a, b, mid): 

    f_a = math.exp(-a) - math.sin(2*a) 
    f_mid = math.exp(-mid) - math.sin( 2*mid) 
    f_b = math.exp(-b) - math.sin(2* b) 

    print("I")


    if abs(f_mid)  < 0.00005:
        return mid

    if f_a * f_mid < 0: 

        new_mid_x = (a + mid)/2 
        res = Bisection(a, mid, new_mid_x)
        return res 
        
    else:
        #print("0" *20)
        new_mid_x = (b + mid)/2 
        res = Bisection(mid, b, new_mid_x )
        
        return res 


def MRF_1(a,b):
    f_a = math.exp(-a) - math.sin(2*a) 
    f_b = math.exp(-b) - math.sin(2* b)
    w_0 = a 
    f_w0 = math.exp(-w_0) - math.sin( 2*w_0) 
    count =1
    
    while True: 
        #print('I')
        w = (f_b * a - f_a *b)/(f_b -f_a)
        f_w1 = math.exp(-w) - math.sin( 2*w) 
        count+=1 
        if f_a * f_w1 <=0: 
            a = a 
            b = w 
            
            f_b = f_w1
            if f_w0* f_w1 >0 :
                f_a = f_a/2 
        else: 
            a = w 
            b = b 
            f_a = f_w1 
            if f_w0* f_w1 >0 :
                f_b = f_b/2 
                
        if abs(f_w1) < 0.00005: 
            break
    return w, count 

def Newton(a): 
    count=1
    while True: 
        #print(count)
        a  = a - (math.exp(-a) - math.sin(2*a))/(-math.exp(-a) - 2*math.cos(2*a))
        f_a = math.exp(-a) - math.sin(2*a) 
        count+=1
        if abs(f_a) < 0.00005: 
            break
    
    return a, count
    

def Scant(a,b):
    f_a = math.exp(-a) - math.sin(2*a) 
    f_b = math.exp(-b) - math.sin(2* b)
    a_1 = a - f_a* (a - b)/(f_a - f_b) 
    count =1 
    
    while True: 
        f_a = math.exp(-a) - math.sin(2*a) 
        f_b = math.exp(-b) - math.sin(2* b)
        a_1 = a - f_a* (a - b)/(f_a - f_b) 
        f_a_1 = math.exp(-a_1) - math.sin(2*a_1) 
        count+=1 
    
        if f_a * f_a_1 <=0: 
            a = a 
            b = a_1 
        else: 
            a = a_1 
            b = b 

        if abs(f_a_1) < 0.00005: 
            break
        
    return a_1, count 
    
    
if __name__ == '__main__':
    
    a = 1 
    b = 2 
    print("=" *20)
    print("Bisection")
    res = Bisection(a, b, (a+b)/2)
    print("result", res )
    print("count  Bisection", 13 )
    
    print("=" *20)
    print("Modified Regular Falsi")
    mrf, c  = MRF_1(a,b,)
    print("result MRF", mrf )
    print("count", c )
    print("=" *20)
    print("Secant")
    Scant, count_s = Scant(a,b)
    print("result Scant", Scant )
    print("count", count_s )
    print("=" *20)
    print("Newton")
    newton, count_n  = Newton(b)
    print("result newton", newton )
    print("count", count_n )


