import numpy as np
from typing import Callable


def gradient_descent(start: float, gradient: Callable[[float], float],
                     learn_rate: float, max_iter: int, tol: float = 0.01):
    x = start
    steps = [start] 

    for _ in range(max_iter):
        diff = learn_rate*gradient(x)
        if np.abs(diff) < tol:
            break
        x = x - diff
        steps.append(x) 
  
    return steps, x

def func1(x:float):
    return x**2-4*x+1

def gradient_func1(x:float):
    return 2*x - 4