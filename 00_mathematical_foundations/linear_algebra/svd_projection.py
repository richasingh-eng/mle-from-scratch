import numpy as np

def compute_svd(X, full_matrices=False):
    return np.linalg.svd(X, full_matrices=full_matrices)

def reconstruct_low_rank(U, S, Vt, k):
    Uk=U[:, :k]
    Sigmak=np.diag(S[:k])
    Vtk=Vt[:k, :]
    return Uk @ Sigmak @ Vtk

def explained_variance_ratio(S, k):
    energy=S ** 2
    return np.sum(energy[:k])/ np.sum(energy)

def compute_reconstruction_error(X, X_renew):
    return np.linalg.norm(X- X_renew, ord='fro')

def project_subspace(X, Vt, k):
    Vk = Vt[:k, :].T
    return X @ Vk
    
# Singular Value Decomposition Implementation
def run_svd_core():
    np.random.seed(42)
    m, d=6, 4
    k=4
    X=np.random.randn(m, d) # X is matrix of shape 6X4
    
    print(f"X shape: {X.shape}")
    
    U, S, Vt= compute_svd(X, full_matrices=False) # np.linalg.svd() decomposes matrix into (U, Sigma, Vt) and full_matrices=False ensures that U, Sigma and Vt doesn't have same size to that of X
       
    print(f"U shape: {U.shape}") #mXk
    print(f"S shape: {S.shape}") #k
    print(f"Vt shape: {Vt.shape}") #kXd
    
    X_proj=project_subspace(X, Vt, k)
    print(f"Projected coordinates (rank-{k}) shape: {X_proj.shape}")
    
    X_low_rank=reconstruct_low_rank(U, S, Vt, k)
    print(f"Reconstructed ambient shape: {X_low_rank.shape}")
    
    error_k=compute_reconstruction_error(X, X_low_rank) # used to find the error
    theoretical_error=np.sqrt(np.sum(S[k:] ** 2))
    variance_retained= explained_variance_ratio(S, k)
    print(f"Rank-{k} Frobenius error: {error_k:.4f}")
    print(f"Theoretical Eckart-Young error: {theoretical_error:.4f}")
    print(f"Energy preserved: {variance_retained * 100:.2f}%")
    print(f"Errors match: {np.isclose(error_k, theoretical_error)}")
    

if __name__=='__main__':
    run_svd_core()
    
    
    