import torch
import threading
import logging

logging.basicConfig(format='%(levelname)s:%(message)s',level=logging.DEBUG)
class PolyGAN_CP_Layer(torch.nn.Module):
    def __init__(self, N, rank, imwidth, imheight, verbose = 0):
        super(PolyGAN_CP_Layer, self).__init__()

        # N = order of the polynomial = order of the tensor A:
        # A is of dimension (s, s, ..., s) = (N x s)
        # rank = rank used for the tensor cores
        # size = length of the flattened input
        self.N = N
        self.rank = rank
        self.s = imwidth * imheight
        self.verbose = verbose

        # bias and weights
        b = torch.empty(self.s,1)
        torch.nn.init.xavier_normal_(b)
        self.b = torch.nn.Parameter(b.reshape(self.s))

        self.W = torch.nn.ParameterList()
        self.shapelist = []
        # 0th order: bias [s], 1st order: weight [s, s], 2nd order: weight [s, s, s], ...
        for n in range(1, N + 1):
            tshape = [self.s]*(n + 1)
            self.shapelist.append(tshape)

            # Here we allocate all the factor matrices (using CP):
            for o in range(n + 1):
                # We need one factor matrix per dimension of the weight matrix of that order.
                # So 3rd order [d, d, d] has 3 factor matrices of shape [d, rank]
                factor_matrix = torch.zeros((self.s, self.rank))
                torch.nn.init.xavier_normal_(factor_matrix)
                self.W.append(torch.nn.Parameter(factor_matrix))

        if self.verbose != 0:
            print("self.shapelist =")
            for shape in self.shapelist:
                print(shape)
            print("self.N =", self.N, "?=", len(self.shapelist), "= len(self.shapelist)")
            print("W has", len(self.W), "elements of shape", self.W[0].shape)


    def parallelScalarProd(self, n, r):
        f = 1
        for k in range(1, n):
            f *= torch.dot(self.z, self.W[k][:,r])
        # The .data was necessary to remove gradient info, as the operation was listed as in-place
        self.VecProds.data[n, r] = f
        if self.verbose != 0:
            logging.info('f = %f, is a single number', f)
        return

    def parallelRankSum(self, n):
        threads = []
        for r in range(self.rank):
            # Perform parallel computation of the products: tremendous speedup.
            process = threading.Thread(target=self.parallelScalarProd, args=(n, r,))
            process.start()
            threads.append(process)

        Rsum = torch.zeros(self.s)
        for r in range(self.rank):
            threads[r].join()
            Rsum += self.W[0][:,r] * self.VecProds[n, r]
        self.Rsums[n] = Rsum
        if self.verbose != 0:
            shapes = list(self.Rsums[n].shape)
            logging.info('Size of Rsum[%d] = %d', n, shapes[0])
        return

    def forward(self, z):
        # Compute the forward pass: the nmode multiplications f = self.b + self.b + self.W[0]*z + z.T*...
        # We can also compute every rank computation in parallel and sum results.
        # For every n, spawn a thread to compute one part of the Nsum, so Nsum = sum(Rsums).
        # For every r, spawn a thread to compute one part of the Rsum, so Rsum = sum(a^1*VecProds)
        self.z = z
        threads = []
        self.Rsums = torch.zeros(self.N, self.s) #[None]*self.N
        self.VecProds = torch.zeros(self.N, self.rank) #[None]*self.rank*self.N
        for n in range(self.N):
            # Perform parallel computation of the rank summation: tremendous speedup.
            process = threading.Thread(target=self.parallelRankSum, args=(n,))
            process.start()
            threads.append(process)

        Nsum = torch.zeros(self.s)
        for n in range(self.N-1, -1, -1):
            threads[n-1].join()
            Nsum += self.Rsums[n-1]

        return Nsum + self.b


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
            #TTcore /= self.s
            self.TT.append(torch.nn.Parameter(TTcore))

        if self.verbose != 0:
            print("self.ranklist =", self.ranklist)
            print("self.N =", self.N, "+ 1 ?=", len(self.ranklist), "= len(self.ranklist)")
            print("TT has", len(self.TT), "elements:")
            for i, tt in enumerate(self.TT):
                print("element", i, ":", tt.shape)

    def test(self):
        # The multiplication z_i * G_k(i) is really [1] * [rank, rank] (or for last cart [1] * [rank, 1] )
        # Here we test whether the vectorized multiplication yields the same result. Warning: this is slow!
        k = 1
        # According to the algorithm:
        summation = 0
        for i in range(self.s):
            summation += self.z[i] * self.TT[k][:,i,:]

        # Naive fast implmentation is: self.V[k] = self.z @ self.TT[k]. Does not yield same results.
        # Instead, permute the axes of the train cart from [rank, s, rank] to [s, rank**2] (or [s, rank*1])
        perm = self.TT[k].permute(1,0,2).reshape(self.s,-1)
        product = torch.matmul(self.z, perm).reshape(self.ranklist[k], self.ranklist[k+1])

        # Assert whether they are the same:
        print("Are the two methods equivalent?", torch.allclose(summation, product, rtol = 1e-03, atol = 1e-03))
        return

    def parallelVecProd(self, k):
        d1, d2 = self.ranklist[k], self.ranklist[k+1]
        tmp = torch.matmul(self.z, self.TT[k].permute(1,0,2).reshape(self.s, d1*d2))
        self.V.data[k,0:d1,0:d2] = tmp.reshape(d1, d2)
        return

    def forward(self, z):
        # Compute the forward pass: the nmode multiplications f = A x1 z x2 z x3 ··· x(N-1) z
        # Follow algorithm 1: allocate space and compute each V^(k), possible in parallel with threads.
        # V^(0) will not be used as it is unnecessary to compute it, it's just self.TT[0][0,:,:]
        self.V = torch.zeros(self.N, self.rank, self.rank) #[None] * self.N
        self.z = z
        threads = []

        if self.verbose != 0:
            print("Threads are computing..")
            self.test()

        for k in range(1, self.N):
            # Perform parallel computation of the products: tremendous speedup. See test() for more info.
            process = threading.Thread(target=self.parallelVecProd, args=(k,))
            process.start()
            threads.append(process)

        # Start the whole product chain now, so that we have [(1),s,r] x [r, r] x ... x [r, (1)] = s
        f = self.TT[0][0,:,:]
        for k in range(1, self.N):
            d1, d2 = self.ranklist[k], self.ranklist[k+1]
            threads[k-1].join()
            f @= self.V[k,0:d1,0:d2]
        return f.reshape(-1)


class Generator(torch.nn.Module):
    def __init__(self, layer, batch_size, N, rank, imwidth, imheight, scalefactor):
        super(Generator, self).__init__()

        self.batch_size = batch_size
        self.c = 1
        self.imwidth, self.imheight = imwidth, imheight
        self.s = imwidth*imheight
        self.PolyLayer = layer(N, rank, imwidth, imheight, 0)
        # Register the scalefactor of the upscaling procedure in the forward pass of the generator.
        self.scalefactor = scalefactor

    def BatchInParallel(self, b):
        # self.c-1 because for one channel we want to access index 0
        self.x[b, self.c-1, :] = self.PolyLayer(self.x[b, self.c-1, :])
        return

    def forward(self, x):
        # Register x as attribute for parallel access
        self.x = x

        # Pytorch implementation = better:
        upscale = torch.nn.Upsample(scale_factor=self.scalefactor, mode='bilinear', align_corners=False)
        self.x = upscale(self.x)

        self.x = self.x.reshape(self.batch_size, self.c, self.s) # flatten to the 1D equivalent vector

        # Start threads
        threads = []
        for batch in range(self.batch_size):
            # Perform parallel computation of the rank summation: tremendous speedup.
            process = threading.Thread(target=self.BatchInParallel, args=(batch,))
            process.start()
            threads.append(process)

        # Wait for threads to end:
        for batch in range(self.batch_size):
            threads[batch].join()

        #self.x = self.PolyLayer(self.x)
        self.x = self.x.reshape(self.batch_size, self.c, self.imwidth, self.imheight)
        return self.x
