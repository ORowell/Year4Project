import timeit
from typing import Union

def zero_bessel_time():
    setup = """
from scipy.special import kn
import numpy as np

ary = np.zeros(1000)
    """
    
    test = 'kn(1, ary)'
    
    return min(timeit.repeat(test, setup, number=10000))

def random_bessel_time():
    setup = """
from scipy.special import kn
import numpy as np

ary = np.random.uniform(9, 20, 1000)
    """
    
    test = 'kn(1, ary)'
    
    return min(timeit.repeat(test, setup, number=10000))

def random_bessel_time_k1():
    setup = """
from scipy.special import k1
import numpy as np

ary = np.random.uniform(9, 20, 1000)
    """
    
    test = 'k1(ary)'
    
    return min(timeit.repeat(test, setup, number=10000))

# print(zero_bessel_time())
# print(random_bessel_time())
# print(random_bessel_time_k1())

import numpy as np
from scipy.special import kn
A_X = 1
LAMBDA = 1

def v_l_true(x, y, N=100):
    total = 0
    for n in range(-N, N+1):
        vals = np.sqrt((x + n*A_X)**2 + y**2)/LAMBDA
        total += kn(0, vals)
    return total

def v_l_analytic(x, y, N=100):
    total = 0
    for n in range(-N, N+1):
        const = 2*np.pi*n/A_X
        q_n = np.sqrt(1/LAMBDA**2 + const**2)
        vals = np.exp(-1j*const*x - q_n*np.abs(y))/q_n
        total += vals
    return np.pi/A_X*total

def v_l_fourier_analytic(kx, ky, N=100):
    total = 0
    const = 1/LAMBDA**2 + kx**2 + ky**2
    for n in range(-N, N+1):
        val = np.exp(-1j*n*A_X*kx)
        total += val/const
    return total

    
# n = 10000
# x=np.linspace(0, 15, 100)
# a = v_l_true(x, 1, n)
# b = v_l_analytic(x, 1, n)
# print(a-b)


def zero_size_norm_time():
    setup = """
import numpy as np

ary = np.empty(shape=(0,2))
    """
    
    test = 'np.linalg.norm(ary, axis=1)'
    
    return min(timeit.repeat(test, setup, number=int(1e6)))

def one_size_norm_time():
    setup = """
import numpy as np

ary = np.random.uniform(0, 10, (1, 2))
    """
    
    test = 'np.linalg.norm(ary, axis=1)'
    
    return min(timeit.repeat(test, setup, number=int(1e6)))

def one_size_simple_norm_time():
    setup = """
import numpy as np

ary = np.random.uniform(0, 10, (1, 2))
    """
    
    test = 'np.sum(ary**2)**0.5'
    
    return min(timeit.repeat(test, setup, number=int(1e6)))

def big_size_norm_time():
    setup = """
import numpy as np

ary = np.random.uniform(0, 10, (100, 2))
    """
    
    test = 'np.linalg.norm(ary, axis=1)'
    
    return min(timeit.repeat(test, setup, number=int(1e6)))

print(zero_size_norm_time())        # 9.163629099999998     or 9.239294699999995
print(one_size_norm_time())         # 8.4053362             
print(one_size_simple_norm_time())  # 7.8467337999999955    or 8.624127400000006
print(big_size_norm_time())         # 17.320417900000024    or 11.606690099999994