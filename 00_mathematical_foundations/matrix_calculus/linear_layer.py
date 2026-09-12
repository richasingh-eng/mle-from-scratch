import sys
from pathlib import Path

import numpy as np

# Ensure Python can find grad_checker when running from repo root
sys.path.append(str(Path(__file__).resolve().parent))
from grad_checker import check_gradient, eval_numerical_gradient
def linear_forward(X, w, b):
    # X is a matrix of size mXd
    # w is a matrix of size dXn
    # b is a vector of size (m,)
    out=X@w+b # out will have a size mXn
    cache=(X, w, b)
    return out, cache

def linear_backward(dout, cache):
    X, w, b=cache
    dX=dout @ w.T
    
    dw=X.T @ dout
    
    db=np.sum(dout, axis=0)
    
    return dX, dw, db

if __name__=='__main__':
    np.random.seed(42)
    N, D, M=3, 4, 2
    x=np.random.randn(N,D)
    w=np.random.randn(D,M)
    b=np.random.randn(M)
    dout=np.random.randn(N,M)
    
    out, cache=linear_forward(x, w, b)
    dx, dw, db=linear_backward(dout, cache)
    
    f_x=lambda x_val:np.sum(linear_forward(x_val, w, b)[0]*dout)
    f_w=lambda w_val:np.sum(linear_forward(x, w_val, b)[0]*dout)
    f_b=lambda b_val:np.sum(linear_forward(x, w, b_val)[0]*dout)
    
    print("Checking dx:")
    check_gradient(f_x, x, dx)

    print("Checking dw:")
    check_gradient(f_w, w, dw)

    print("Checking db:")
    check_gradient(f_b, b, db)

    print("\nAll linear layer gradient checks passed!")