import torch

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
            #torch.nn.init.xavier_uniform_(TTcore)
            TTcore /= self.s
            self.TT.append(torch.nn.Parameter(TTcore))

    def forward(self, z):
        self.z = z.clone()

        f = self.TT[0][0]
        for k in range(1, self.N):
            d1, d2 = self.ranklist[k], self.ranklist[k+1]

            extra = torch.matmul(self.z, self.TT[k].permute(1,0,2).reshape(self.s, d1*d2))
            extra = extra.reshape(d1, d2)
            f = torch.matmul(f, extra)

        out = f.reshape(-1)
        return out


class Generator(torch.nn.Module):
    def __init__(self, layer, N, rank, imwidth, imheight,scalefactor):
        super(Generator, self).__init__()

        self.c = 1
        self.imwidth, self.imheight = imwidth, imheight
        self.s = imwidth*imheight
        self.PolyLayer = layer(N, rank, imwidth, imheight, 0)
        self.BN = torch.nn.BatchNorm2d(num_features=1)
        self.upsample = torch.nn.Upsample(scale_factor=scalefactor, mode='bilinear', align_corners=False)

    def forward(self, x):
        # Register dimensions:
        xshape = x.shape
        if len(xshape) == 2:
            batch_size = 1
        elif len(xshape) == 4:
            batch_size = xshape[0]
        else:
            raise ValueError

        # Register x as attribute for parallel access, and clone because dataset would be overwritten
        self.x = self.BN(x.clone())
        self.x = self.upsample(self.x)

        #print('x.shape = ', x.shape)
        self.x = self.x.reshape(batch_size, self.c, self.s) # flatten to the 1D equivalent vector

        for batch in range(batch_size):
            self.x[batch, self.c-1, :] = self.PolyLayer(self.x[batch, self.c-1, :])

        self.x = self.x.reshape(batch_size, self.c, self.imwidth, self.imheight)
        return self.x
