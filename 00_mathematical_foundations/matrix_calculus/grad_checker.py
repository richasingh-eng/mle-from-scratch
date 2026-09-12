import numpy as np

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
        
        x[idx]=orig_val
        
        grad[idx]=(fx_plus-fx_minus)/(2*eps)
        it.iternext()
        
    return grad

def check_gradient(f, x, analytic_grad):
        num_grad=eval_numerical_gradient(f, x)
        rel_error=np.max(np.abs(analytic_grad-num_grad)/
                         (np.maximum(np.abs(analytic_grad), np.abs(num_grad))+1e-15))
        print(f"Relative error: {rel_error:.2e}")
        assert rel_error < 1e-5, "Gradcheck failed!"
if __name__ == "__main__":
    # Test function: f(x) = sum(x^2)
    f = lambda x: np.sum(x**2)
    
    # Dummy input
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    
    # Analytical gradient: d/dx (x^2) = 2x
    analytic_grad = 2 * x
    
    check_gradient(f, x, analytic_grad)

        