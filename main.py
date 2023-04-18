# Import the FastAPI library
from fastapi import FastAPI

# Create an instance of the FastAPI class
app = FastAPI()

from numba import jit
import numpy as np

x = np.arange(100).reshape(10, 10)

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def go_fast(a): # Function is compiled to machine code when called the first time
    trace = 0.0
    
    for i in range(a.shape[0]):   # Numba likes loops
        trace += np.tanh(a[i, i]) # Numba likes NumPy functions
    return a + trace              # Numba likes NumPy broadcasting


# Define a route for the root URL ("/") using the get() decorator
@app.get("/")
def read_root():
    x = np.arange(100).reshape(10, 10)
    res = sum(sum(go_fast(x)))
    return {"message": f"Hello, {res}!"}

# Define another route with a path parameter
@app.get("/greet/{name}")
def greet(name: str):
    return {"message": f"pizdtes, {name}!"}


if  __name__=='__main__':
    
    print('hello')