import torch
import threading
import logging

logging.basicConfig(format='%(levelname)s:%(message)s',level=logging.DEBUG)

# Sequential version:
class FTT_Layer(torch.nn.Module):
    def __init__(self, N, rank, imwidth, imheight, verbose = 0):
        super(FTT_Layer, self).__init__()

        # N = order of the polynomial = order of the tensor A:
        # A is of dimension (s, s, ..., s) = (N x s)
        # rank = rank used for the tensor cores
        # size = length of the flattened input
        self.N = N
        self.rank = rank
        self.s = imwidth * imheight
        self.verbose = verbose
        self.lock = threading.Lock()

        # Make a list of TTcore ranks, starting from r_0 = r_N = 1: perhaps feed it in as a list? isinstance(rank, list)
        self.ranklist = [1, 1]
        for n in range(self.N - 1):
            self.ranklist.insert(-1, self.rank)

        # Start by making the tensor train: store the matrices in one big parameterlist
        self.TT = torch.nn.ParameterList()

        # Make s instances for every mode of the tensor, or make a 3D tensor instead:
        for n in range(self.N):
            # Make tensors of size (r_{k-1}, n_{k} = self.s, r_{k})
            TTcore = torch.empty(self.ranklist[n], self.s, self.ranklist[n+1])
            torch.nn.init.xavier_normal_(TTcore)
            TTcore /= self.s
            self.TT.append(torch.nn.Parameter(TTcore))


    def parallelVecProd(self, k):
        d1, d2 = self.ranklist[k], self.ranklist[k+1]
        cop = self.TT[k].clone()
        tmp = torch.matmul(self.z, cop.permute(1,0,2).reshape(self.s, d1*d2))
        #self.V[k,0:d1,0:d2] = tmp.reshape(d1, d2)
        ####self.V[k] = tmp.reshape(d1, d2)
        #with self.lock:
        #    print('before: shape =', self.V[k,0:d1,0:d2].shape, 'version = ', self.V[k,0:d1,0:d2]._version)

        with torch.no_grad():
            self.V[k,0:d1,0:d2] = tmp.reshape(d1, d2)
        #with self.lock:
        #    print('after: shape =', self.V[k,0:d1,0:d2].shape, 'version = ', self.V[k,0:d1,0:d2]._version)

        return

    def forward(self, z):
        # Compute the forward pass: the nmode multiplications f = A x1 z x2 z x3 ··· x(N-1) z
        # Follow algorithm 1: allocate space and compute each V^(k), possible in parallel with threads.
        # V^(0) will not be used as it is unnecessary to compute it, it's just self.TT[0][0,:,:]
        self.z = z.clone()

        f = self.TT[0][0]
        for k in range(1, self.N):
            d1, d2 = self.ranklist[k], self.ranklist[k+1]
            extra = torch.matmul(self.z, self.TT[k].permute(1,0,2).reshape(self.s, d1*d2))
            extra = extra.reshape(d1, d2)
            f = torch.matmul(f, extra)
        return f.reshape(-1)


class Generator(torch.nn.Module):
    def __init__(self, layer, batch_size, N, rank, imwidth, imheight):
        super(PolyNet, self).__init__()

        self.batch_size = batch_size
        self.c = 1
        self.imwidth, self.imheight = imwidth, imheight
        self.s = imwidth*imheight
        self.PolyLayer = layer(N, rank, imwidth, imheight, 0)
        self.SM = torch.nn.Sigmoid()

    def forward(self, x):
        # Register x as attribute for parallel access
        self.x = x.clone()

        #print('x.shape = ', x.shape)
        self.x = self.x.reshape(self.batch_size, self.c, self.s) # flatten to the 1D equivalent vector

        for batch in range(self.batch_size):
            self.x[batch, self.c-1, :] = self.PolyLayer(self.x[batch, self.c-1, :])

        self.x = self.x.reshape(self.batch_size, self.c, self.imwidth, self.imheight)
        return self.x #self.SM(self.x)
