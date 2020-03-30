# This python file contains the latest functionality constructed by the Polytion team.
# Here, all tensor decomposition functions can be found so far

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import torch
import skimage


def version():
    print("version 1.0 30/03/2020")


def remap(image, low, high):
    OldRange = (np.max(image) - np.min(image))
    NewRange = (high - low)
    return (((image - np.min(image)) * NewRange) / OldRange) + low


def KronProd2D(P, Q):
    # Register dimensions.
    I, J = P.shape
    K, L = Q.shape

    # Adjust dimensions of P and Q to perform smart multiplication:
    # interweave the dimensions containing values and perform elementwise multiplication.
    P = P.view(I, 1, J, 1)
    Q = Q.view(1, K, 1, L)

    R = P * Q
    return R.view(I*K, J*L)


def KronProd(P, Q):
    # This should work for higher order tensors.
    # Register and check dimensions.
    pshape = P.shape
    qshape = Q.shape
    if P.dim() != Q.dim():
        raise ValueError('Matrices should be of the same order: P.dim() =' + str(P.dim()) + ' != Q.dim() = ' + str(Q.dim()))

    # Adjust dimensions of P and Q to perform smart multiplication:
    # interweave the dimensions containing values and perform elementwise multiplication.
    # Start with a list of ones and set dimensions as every even or uneven index.
    pindices = [1]*2*len(pshape)
    pindices[::2] = pshape
    qindices = [1]*2*len(qshape)
    qindices[1::2] = qshape

    P = P.view(pindices)
    Q = Q.view(qindices)

    R = P * Q
    rshape = [p*q for p, q in zip(pshape,qshape)]
    return R.view(rshape)


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
        R[:,j] = KronProd(P[:,j], Q[:,j])

    # Tried with slicing but did not seem to work:
    # column_indices = list(range(J)) or slice(0,J)
    # R[:, column_indices] = KronProd(P[:, column_indices], Q[:, column_indices])
    return R


def Permute_tensor(tensor, origin, axis, outputshape=None):
    # Perform the permutation of the tensor here, to be used in both Fold and Unfold.
    # For more explanation on this function see Unfold().

    dims = list(range(tensor.dim()))
    # For later use, we want to be able to map indices for a circular permutation
    if origin < 0: origin = dims[origin]
    if axis < 0: axis = dims[axis]

    dims.pop(origin)
    dims = dims[::-1]
    dims.insert(axis, origin)
    tensor = tensor.permute(*dims)

    if outputshape:
        tensor = tensor.reshape(outputshape)
    return tensor


def Unfold(tensor, mode):
    # Notation sets minimal mode to be 1, but we want to map that to zero. We also cannot mode_n unfold a matrix with n-1 modes.
    if mode > len(tensor.shape) or mode < 0:
        raise ValueError('Tensor has order ' + str(len(tensor.shape)) + ', cannot give mode ' + str(mode))

    # Register tensor shape and mode size
    order = tensor.dim()
    tshape = tensor.shape
    I_n = tshape[mode]

    # To get the prescribed matrix in Kolda et al, we need to perform a permutation of the axes.
    # The selected mode needs to be viewed as the first, and all other dimensions should reverse their order.
    # It took a lot of experiments to get this right, the full functionality has been moved to Permute_tensor().
    return Permute_tensor(tensor, mode, 0, (I_n, -1))


def Fold(unfolded_tensor, mode, desired_shape):
    # Perform the same entry safety controls
    if mode > len(desired_shape) or mode < 0:
        raise ValueError('Folded tensor has order ' + str(len(desired_shape)) + ', cannot give mode ' + str(mode))

    # Before calling Permute_tensor(), we need to reshape the unfolded tensor back to its' desired order.
    # This does not yield the desired tensor but takes into account the coming permutation.
    shape = list(desired_shape)
    axis = shape.pop(mode)
    shape = shape[::-1]
    shape.insert(0, axis)
    tensor = unfolded_tensor.reshape(shape)

    # return Permute_tensor(tensor, 0, mode, desired_shape)
    # Exactly my point before, we have accounted for the exact inverse transformation without the need for a reshape.
    # Both return statements yield the same result.
    return Permute_tensor(tensor, 0, mode)


def CPD(tensor, rank=5, maxiter=100):
    # Problems were recorded with tensors that had only type = torch.LongTensor.
    # Reformat into torch.FloatTensor, and clone to avoid performing CPD twice on the same tensor.
    tensor = tensor.type(torch.FloatTensor).clone()

    # Note that we will perform n_iter_max iterations per tensor order!
    # Register input variables:
    tshape = tensor.shape
    order = tensor.dim()

    # Initialization: create a list of matrices to represent the input tensor: tensor = [t_1, t_2, .. , t_order]
    # Incorrect: ABC = [torch.rand((rank, rank))]*order
    ABC = []
    for o in range(order):
        ABC.append(torch.rand((tshape[o], rank)))

    # Main loop: for every iteration, adjust every factor matrix
    for iteration in range(maxiter):
        for i, fmatrix in enumerate(ABC):
            # Get all the other fmatrices, temporarily remove the element:
            ABC.pop(i)

            # Construct V: use the first element ABC[0] and the loop over everything after with ABC[1::]
            V = ABC[0].t() @ ABC[0]
            # pinverse_V = torch.ones((rank, rank), dtype=tensor.dtype)
            for ABC_notfmatrix in ABC[1::]:
                V = V * (ABC_notfmatrix.t() @ ABC_notfmatrix)
                # pinverse_V = pinverse_V * (fmatrix.t() @ fmatrix)

            # Construct W (kr product of all matrices other way around):
            # take the last element of ABC and then loop over ABC[::-1] excluding the last element.
            # Example: ABC = [1,2,3,4] and ABC[::-1][1::] = [3,2,1] and ABC[1::-1] = [2,1]
            # This would be easier with a pop and insert, but we need to recycle ABC a lot...
            W = ABC[-1]
            for ABC_notfmatrix in ABC[::-1][1::]:
                W = KathRaoProd(W, ABC_notfmatrix)

            # Here is a problem:
            fmatrix = Unfold(tensor, i) @ W @ torch.pinverse(V)
            # Other tries to get the p_inverse
            # pinverse_V = ((V.t() @ V).inverse()) @ V.t()
            # pinverse_V = V.t() @ ((V.t() @ V).inverse())
            # fmatrix = Unfold(tensor, i) @ W @ pinverse_V

            # Push the lost fmatrix back in:
            ABC.insert(i, fmatrix)
    return ABC


def UnfoldFM(factor_matrices, mode):
    if mode > len(factor_matrices) or mode < 0:
        raise ValueError('Tensor has order ' + str(len(factor_matrices)) + ', cannot give mode ' + str(mode))
    # Start with the first element (for normal dot product), then the last element is ready for the backward loop.
    first = factor_matrices.pop(mode)
    prod = factor_matrices.pop(-1)
    for factor in factor_matrices[::-1]:
        prod = KathRaoProd(prod, factor)
    return first @ prod.t()


def Compose(factors):
    # Make copy to remove dependancy
    factor_matrices = []
    for f in factors:
        factor_matrices.append(f.clone())

    # Register dimensions of the tensor:
    order = len(factor_matrices)
    tshape = [x.shape[0] for x in factor_matrices]

    # Use the mode-0 unfolding and just fold it back.
    return Fold(UnfoldFM(factor_matrices, 0), 0, tshape)


class PolytionLayer(torch.nn.Module):
    def __init__(self, N, im_w, im_h):
        super(PolyLayer, self).__init__()

        self.d = im_w * im_h
        # N = order of the polynomial
        # d = length of the flattened input

        # Currently arvitrary rank choice
        self.rank = 3
        self.shapes = []

        self.b = torch.nn.Parameter(torch.zeros((self.d)).normal_(0, np.sqrt(2)))
        self.W = torch.nn.ParameterList()

        # 0th order: bias [d], 1st order: weight [d, d], 2nd order: weight [d, d, d]
        for n in range(1, N + 1):
            tshape = [self.d]*(n + 1)
            self.shapes.append(tshape)

            # Here we allocate all the factor matrices (using CP):
            for o in range(n + 1):
                # We need one factor matrix per dimension of the weight matrix of that order.
                # So 3rd order [d, d, d] has 3 factor matrices of shape [d, rank]
                factor = torch.zeros((self.d, self.rank)).normal_(0, np.sqrt(2))
                self.W.append(torch.nn.Parameter(factor))

    def polyGAN(self, data):
        #return self.b + self.W[0]*data + data.T*...
        summation = self.b
        Wcounter = 0
        for n in range(1, N + 1):
            # n = 1: first  order so 2D matrix with 2 fm's
            # n = 2: second order so 3D tensor with 3 fm's
            allprod = torch.ones(self.d)
            for k in range(n):
                vecprod = 0
                for r in range(self.rank):
                    fm_col = self.W[k][:,r]
                    vecprod += data * fm_col
                allprod *= vecprod
        return summation

    def forward(self, x):
        return self.polyGAN(x)


class PlytionNet(torch.nn.Module):
    def __init__(self, N, im_w, im_h):
        super(Net, self).__init__()

        self.im_w, self.im_h = im_w, im_h
        self.PG = PolyLayer(N, im_w, im_h)

    def forward(self, x):
        # Find bilinear version of this:
        x = skimage.transform.resize(x, (self.im_w, self.im_h), anti_aliasing=True)
        x = torch.tensor(x) # make tensor
        x = x.reshape(self.im_w*self.im_h) # flatten to the 1D equivalent vector
        x = x.float() # no need for the very high precision
        return self.PG(x)
