import numpy as np

def compute_svd(X, full_matrices=False):
    return np.linalg.svd(X, full_matrices=full_matrices)

def reconstruct_matrices(U, S, Vt):
    Sigma=np.diag(S)
    return U@Sigma@Vt

def project_subspace(X, Vt, k):
    Vk=Vt[:k, :].T
    return X@Vk

def compute_reconstruction_error(X, X_renew):
    return np.linalg.norm(X- X_renew, ord='fro')
    
# Singular Value Decomposition Implementation
def run_svd_core():
    np.random.seed(42)
    m, d=6, 4
    X=np.random.randn(m, d) # X is matrix of shape 6X4
    
    print(f"X shape: {X.shape}")
    
    U, S, Vt= compute_svd(X, full_matrices=False) # np.linalg.svd() decomposes matrix into (U, Sigma, Vt) and full_matrices=False ensures that U, Sigma and Vt doesn't have same size to that of X
    k=len(S)
    
    print(f"U shape: {U.shape}") #mXk
    print(f"S shape: {S.shape}") #k
    print(f"Vt shape: {Vt.shape}") #kXd
    
    X_renew=reconstruct_matrices(U, S, Vt)
    print(f"X reconstructed shape: {X_renew.shape}") # mXd
    
    renew_error=compute_reconstruction_error(X, X_renew) # used to find the error
    print(f"Frobenius Norm || X-U Sigma Vt || _F: {renew_error:.2e}")
    print(f"Exact match within tolerance: {np.allclose(X, X_renew)}") # Returns True if two matrices are element wise equal within limited tolerance
    

if __name__=='__main__':
    run_svd_core()
    
    
    