import numpy as np
import torch

P = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
Q = torch.tensor([[1, 4, 7], [2, 5, 8], [3, 6, 9]])

kron=np.kron(P, Q)

def KathRaoProd(P, Q):
    # Register and check dimensions
    pshape = P.shape
    qshape = Q.shape
    J = pshape[-1]
    if J != qshape[-1]:
        raise ValueError('Matrices should have the same number of columns: ' + str(J) + ' != ' + str(qshape[-1]))
    
    # Make R an empty tensor
    rshape = [p*q for p, q in zip(pshape,qshape)]
    rshape[-1] = J
    R = torch.zeros(rshape)
    for j in range(J):        
        R[:,j] = torch.tensor(np.kron(P[j], Q[j]))
        
    R.narrow(0, 0, 2)
        
    # Tried with slicing but did not seem to work:
    # column_indices = list(range(J)) or slice(0,J)
    # R[column_indices] = KronPr    od(P[column_indices], Q[column_indices])
    return R

R=KathRaoProd(P, Q)
