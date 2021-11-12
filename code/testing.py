import timeit

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

print(zero_bessel_time())
print(random_bessel_time())
