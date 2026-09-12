import numpy as np

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

def eval_numerical_gradient(f, x, eps=1e-5):
    grad=np.zeros_like(x)
    it=np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx=it.multi_index
        orig_val=x[idx]
        
        x[idx]=orig_val+eps
        fx_plus=f(x)
        
        x[idx]=orig_val-eps
        fx_minus=f(x)
        
        grad[idx]=(fx_plus-fx_minus)/(2*eps)
        x[idx]=orig_val
        it.iternext()
    return grad

def rel_error(x, y):
    return np.max(np.abs(x-y) / (np.maximum(np.abs(x), np.abs(y))+1e-5))

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
    
    dx_num=eval_numerical_gradient(f_x, x)
    dw_num=eval_numerical_gradient(f_w, w)
    db_num=eval_numerical_gradient(f_b, b)
    
    print("Testing Linear Layer Gradients:")
    print(f"dx error: {rel_error(dx, dx_num):.2e}")
    print(f"dw error: {rel_error(dw, dw_num):.2e}")
    print(f"db error: {rel_error(db, db_num):.2e}")
    
    assert rel_error(dx, dx_num) < 1e-7, "dx check failed!"
    assert rel_error(dw, dw_num) < 1e-7, "dw check failed!"
    assert rel_error(db, db_num) < 1e-7, "db check failed!"
    print("\nAll gradient checks passed successfully!")
    
    
    

    