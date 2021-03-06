import torch
import threading
import queue

import pickle 

from math import sqrt, factorial

class PolyLayer(torch.nn.Module):
    def __init__(self, N, rank, s, randnormweights=True, parallel=True):
        # Start inherited structures:
        torch.nn.Module.__init__(self)
        # N = order of the polynomial = order of the tensor A:
        # A is of dimension (s, s, ..., s) = (N x s)
        # rank = rank used for the tensor cores
        # size = length of the flattened input
        # randnormweights = whether to use random weight initialization
        # normalize = whether to divide each weight by size self.s, as suggested by Morten
        self.N = N
        self.rank = rank
        self.s = s
        self.weightgain = 1.0
        # Necessary for threads:
        self.parallel = parallel
        if randnormweights:
            self.initweights = torch.nn.init.xavier_normal_
            
        else:
            self.initweights = torch.nn.init.xavier_uniform_
            
        self.QueueList = [queue.Queue()]


class PolyLayer_seqOld(torch.nn.Module):
    def __init__(self, N, rank, s, randnormweights=True):
        # Start inherited structures:
        torch.nn.Module.__init__(self)
        # N = order of the polynomial = order of the tensor A:
        # A is of dimension (s, s, ..., s) = (N x s)
        # rank = rank used for the tensor cores
        # size = length of the flattened input
        # randnormweights = whether to use random weight initialization
        # normalize = whether to divide each weight by size self.s, as suggested by Morten
        self.N = N
        self.rank = rank
        self.s = s
        self.weightgain_bias = 2
        self.weightgain_weights = 0.4
        self.Laplace = torch.distributions.laplace.Laplace(0, 0.00005)
        self.eps = 0.07
        # Necessary for threads:
        if randnormweights:
            #self.initweights = torch.nn.init.orthogonal_
            self.initweights = torch.nn.init.xavier_normal_
            #self.initweights = torch.nn.init.kaiming_normal_
            #self.initweights = torch.nn.init.normal_
        else:
            self.initweights = None
            #torch.nn.init.xavier_uniform_
            #self.initweights = torch.nn.init.uniform_
            
class PolyLayer_seq(torch.nn.Module):
    def __init__(self, N, rank, s, initmethod, biasgain, weightgain):
        # Start inherited structures:
        torch.nn.Module.__init__(self)
        # N = order of the polynomial = order of the tensor A:
        # A is of dimension (s, s, ..., s) = (N x s)
        # rank = rank used for the tensor cores
        # size = length of the flattened input
        # randnormweights = whether to use random weight initialization
        # normalize = whether to divide each weight by size self.s, as suggested by Morten
        self.N = N
        self.rank = rank
        self.s = s
        self.weightgain_bias = biasgain
        self.weightgain_weights = weightgain
        self.Laplace = torch.distributions.laplace.Laplace(0, 0.00005)
        self.eps = 0.07
        self.initweights = initmethod



class PolyganCPlayer(PolyLayer):
    def __init__(self, N, rank, imwidth, imheight, layeroptions):
        PolyLayer.__init__(self, N, rank, imwidth*imheight, randnormweights=layeroptions['randnormweights'], parallel=layeroptions['parallel'])

        # Initialize the bias
        b = torch.empty(self.s,1)
        self.initweights(b, self.weightgain)
        if layeroptions['normalize']:
            b /= self.s
        self.b = torch.nn.Parameter(b.reshape(self.s))
        # Initialize the weights
        self.W = torch.nn.ParameterList()
        self.shapelist = []
        # 0th order: bias [s], 1st order: weight [s, s], 2nd order: weight [s, s, s], ...
        # for n in range(1, N + 1):
        #     tshape = [self.s]*(n + 1)
        #     self.shapelist.append(tshape)
        #     for o in range(n + 1):
        #         factor_matrix = torch.zeros((self.s, self.rank))
        #         self.initweights(factor_matrix, self.weightgain)
        #         if layeroptions['normalize']:
        #             factor_matrix /= self.s
        #         self.W.append(torch.nn.Parameter(factor_matrix))
        for n in range(1, N + 1):
            tshape = [self.s]*(n + 1)
            self.shapelist.append(tshape)
            factor_matrix = torch.zeros((self.s, self.rank))
            self.initweights(factor_matrix, self.weightgain)
            if layeroptions['normalize']:
                factor_matrix /= self.s
            self.W.append(torch.nn.Parameter(factor_matrix))

    def forwardInSequence(self, z, queue):
        # Simple and straightforward: see notes on github
        Rsums = queue.get()
        for n in range(self.N):
            f = torch.ones(self.rank)
            for k in range(1, n):
                res = torch.matmul(z, self.W[k])
                f = f * res
            Rsums[n] = torch.matmul(self.W[0], f)
        queue.put(Rsums)
        return

    def forwardInSequenceOld(self, z, queue):
        # Simple and straightforward: see notes on github
        Rsums = queue.get()
        for n in range(self.N):
            partialsum = 0
            for r in range(self.rank):
                f = 1
                for k in range(1, n):
                    f *= torch.dot(z, self.W[k][:,r])
                partialsum += self.W[0][:,r] * f
            Rsums[n] = partialsum
        queue.put(Rsums)
        return

    def parallelRankSum(self, queue, z, n):
        Rsums = queue.get()
        partialsum = 0
        for r in range(self.rank):
            f = 1
            for k in range(1, n):
                f *= torch.dot(z, self.W[k][:,r])
            partialsum += self.W[0][:,r] * f
        Rsums[n] = partialsum
        queue.put(Rsums)
        return

    def forwardInParallel(self, z, queue):
        threads = []
        for n in range(self.N):
            # Perform parallel computation of the rank summation: tremendous speedup.
            currentThread = threading.Thread(target=self.parallelRankSum, args=(queue, z, n,))
            currentThread.start()
            threads.append(currentThread)
        # Wait for threads to end:
        for t in threads:
           t.join()
        #for n in range(self.N-1, -1, -1):
        #    threads[n-1].join()
        #    Nsum += self.Rsums[n-1]
        return

    def forward(self, z, queueindex=0):
        z = z.clone()
        # Put Rsums in the queue to share with threads
        Rsums = torch.zeros(self.N, self.s)
        if queueindex > len(self.QueueList):
            raise ValueError
        currentQueue = self.QueueList[queueindex]
        currentQueue.put(Rsums)

        if self.parallel:
            self.forwardInParallel(z, currentQueue)
        else:
            self.forwardInSequence(z, currentQueue)

        Nsum = torch.zeros(self.s)
        Rsums = currentQueue.get()
        for n in range(self.N-1, -1, -1):
            Nsum += Rsums[n-1]
        return Nsum + self.b


class PolyganCPlayer_seqOld(PolyLayer_seq):
    def __init__(self, N, rank, imwidth, imheight, layeroptions):
        PolyLayer_seq.__init__(self, N, rank, imwidth*imheight, randnormweights=layeroptions['randnormweights'])

        # Initialize the bias
        b = torch.empty(self.s,1)
        self.initweights(b, self.weightgain_bias)
        #torch.nn.init.normal_(b, mean=0.02, std=0.01)
        # if layeroptions['normalize']:
        #     b = b / sqrt(self.s)
        # averagepath = '000001_01_01/average'
        # with open(averagepath, 'rb') as handle:
        #     b = pickle.load(handle)
        
        self.b = torch.nn.Parameter(b.reshape(self.s))
        # Initialize the weights
        self.W = torch.nn.ParameterList()
        for n in range(N+1):
            factor_matrix = torch.zeros(self.s, self.rank)
            self.initweights(factor_matrix, self.weightgain_weights)
            self.W.append(torch.nn.Parameter(factor_matrix/(n+1)))

    def forward(self, z, b):
        Rsums = torch.zeros(b, 1, self.s)
        for n in range(self.N):
            f = torch.ones(b, 1, self.rank)
            for k in range(n+1):
                res = torch.matmul(z, self.W[k+1])
                f = f * res
            Rsums += torch.matmul(f, self.W[0].t()) / factorial(n + 1)
        return Rsums + self.b

class PolyganCPlayer_seq(PolyLayer_seq):
    def __init__(self, N, rank, imwidth, imheight, layeroptions):
        PolyLayer_seq.__init__(self, N, rank, imwidth*imheight, layeroptions["initmethod"], layeroptions["biasgain"], layeroptions["weightgain"])

        # Initialize the bias
        b = torch.empty(self.s,1)
        if layeroptions["initmethod"] == "average":
            averagepath = '000001_01_01/average'
            with open(averagepath, 'rb') as handle:
                b = pickle.load(handle)
            self.initweights = torch.nn.init.xavier_normal_
        else: 
            print(self.initweights, self.weightgain_bias)
            self.initweights(b, self.weightgain_bias)
        self.b = torch.nn.Parameter(b.reshape(self.s))
        # Initialize the weights
        self.W = torch.nn.ParameterList()
        for n in range(N+1):
            factor_matrix = torch.zeros(self.s, self.rank)
            self.initweights(factor_matrix, self.weightgain_weights)
            self.W.append(torch.nn.Parameter(factor_matrix/(n+1)))

    def forward(self, z, b):
        Rsums = torch.zeros(b, 1, self.s)
        for n in range(self.N):
            f = torch.ones(b, 1, self.rank)
            for k in range(n+1):
                res = torch.matmul(z, self.W[k+1])
                f = f * res
            Rsums += torch.matmul(f, self.W[0].t()) / factorial(n + 1)
        return Rsums + self.b



class PolyclassFTTlayer(PolyLayer):
    def __init__(self, N, rank, imwidth, imheight, layeroptions):
        PolyLayer.__init__(self, N, rank, imwidth*imheight, randnormweights=layeroptions['randnormweights'], parallel=layeroptions['parallel'])

        # Make a list of TTcore ranks, starting from r_0 = r_N = 1:
        self.ranklist = [1, 1]
        for n in range(self.N - 1):
            self.ranklist.insert(-1, self.rank)
        # Start by making the tensor train: store the matrices in one big parameterlist
        self.TT = torch.nn.ParameterList()
        for n in range(self.N):
            # Make tensors of size (r_{k-1}, n_{k} = self.s, r_{k})
            TTcore = torch.zeros(self.ranklist[n], self.s, self.ranklist[n+1])
            #self.initweights(TTcore, self.weightgain)
            torch.nn.init.orthogonal_(TTcore)
            if layeroptions['normalize']:
                #TTcore /= sqrt(self.s)
                TTcore /= torch.norm(TTcore)
            self.TT.append(torch.nn.Parameter(TTcore))

    def forwardInSequence(self, z, queue):
        V = queue.get()
        for k in range(1, self.N):
            d1, d2 = self.ranklist[k], self.ranklist[k+1]
            V[k,0:d1,0:d2] = torch.matmul(z, self.TT[k].permute(1,0,2).reshape(self.s, d1*d2)).reshape(d1, d2)
        queue.put(V)
        return

    def parallelVecProd(self, queue, z, k):
        d1, d2 = self.ranklist[k], self.ranklist[k+1]
        V = queue.get()
        V[k,0:d1,0:d2] = torch.matmul(z, self.TT[k].permute(1,0,2).reshape(self.s, d1*d2)).reshape(d1, d2)
        queue.put(V)
        return

    def forwardInParallel(self, z, queue):
        threads = []
        for k in range(1, self.N):
            currentThread = threading.Thread(target=self.parallelVecProd, args=(queue, z, k,))
            currentThread.start()
            threads.append(currentThread)
        # Wait for threads to end:
        for t in threads:
            t.join()
        return

    def forward(self, z, queueindex=0):
        z = z.clone()
        # Put V in the right queue
        V = torch.empty(self.N, self.rank, self.rank)
        if queueindex > len(self.QueueList):
            raise ValueError
        currentQueue = self.QueueList[queueindex]
        currentQueue.put(V)

        if self.parallel:
            self.forwardInParallel(z, currentQueue)
        else:
            self.forwardInSequence(z, currentQueue)

        # Start the whole product chain now, so that we have [(1),s,r] x [r, r] x ... x [r, (1)] = s
        V = currentQueue.get()
        f = self.TT[0][0]
        for k in range(1, self.N):
            d1, d2 = self.ranklist[k], self.ranklist[k+1]
            f = torch.matmul(f, V[k,0:d1,0:d2])
        # By now the current queue should be empty:
        if not currentQueue.empty():
            raise ValueError
        return f.reshape(-1)


class PolyclassFTTlayer_seqOld(PolyLayer_seq):
    def __init__(self, N, rank, imwidth, imheight, layeroptions):
        PolyLayer_seq.__init__(self, N, rank, imwidth*imheight, randnormweights=layeroptions['randnormweights'])

        self.ranklist = [1, 1]
        for n in range(self.N - 1):
            self.ranklist.insert(-1, self.rank)
        # Start by making the tensor train: store the matrices in one big parameterlist
        self.TT = torch.nn.ParameterList()
        for n in range(self.N):
            # Make tensors of size (r_{k-1}, n_{k} = self.s, r_{k})
            TTcore = torch.zeros(self.ranklist[n], self.s, self.ranklist[n+1])
            self.initweights(TTcore, self.weightgain)
            #torch.nn.init.orthogonal_(TTcore, 0.8)
            if layeroptions['normalize']:
                TTcore /= sqrt(sqrt(self.s))
                #TTcore /= torch.norm(TTcore)
            self.TT.append(torch.nn.Parameter(TTcore))

    def forward(self, z, b):
        z = z.clone()
        V = torch.zeros(b, 1, self.N, self.rank, self.rank)
        
        for k in range(self.N):
            d1, d2 = self.ranklist[k], self.ranklist[k+1]
            # print(self.TT[k].shape)
            # print(self.TT[k].permute(1,0,2).reshape(self.s, d1*d2).shape)
            V[:,:,k,0:d1,0:d2] = torch.matmul(z, self.TT[k].permute(1,0,2).reshape(self.s, d1*d2)).reshape(b, 1, d1, d2)

        f = self.TT[0][0]
        for k in range(self.N):
            d1, d2 = self.ranklist[k], self.ranklist[k+1]
            f = torch.matmul(f, V[:,:, k,0:d1,0:d2])
        return f.reshape(-1)

class PolyclassFTTlayer_seq(PolyLayer_seq):
    def __init__(self, N, rank, imwidth, imheight, layeroptions):
        PolyLayer_seq.__init__(self, N, rank, imwidth*imheight, layeroptions["initmethod"], layeroptions["biasgain"], layeroptions["weightgain"])

        self.ranklist = [1, 1]
        for n in range(self.N - 1):
            self.ranklist.insert(-1, self.rank)
        # Initialize the bias
        b = torch.empty(self.s,1)
        if layeroptions["initmethod"] == "average":
            averagepath = '000001_01_01/average'
            with open(averagepath, 'rb') as handle:
                b = pickle.load(handle)
            self.initweights = torch.nn.init.xavier_normal_
        else: 
            self.initweights(b, self.weightgain_bias)
        self.b = torch.nn.Parameter(b.reshape(self.s))
        # Start by making the tensor train: store the matrices in one big parameterlist

        self.TT = torch.nn.ParameterList()
        for n in range(self.N):
            # Make tensors of size (r_{k-1}, n_{k} = self.s, r_{k})
            TTcore = torch.zeros(self.ranklist[n], self.s, self.ranklist[n+1])
            self.initweights(TTcore, self.weightgain_weights)    
            self.TT.append(torch.nn.Parameter(TTcore))

    def forward(self, z, b):
        out = torch.zeros(b, self.s)
        for n in range(self.N):
            #print("n =",n)
            V = torch.zeros(b, 1, self.N, self.rank, self.rank)
            f = self.TT[0][0]
            #print("f1", f.shape)
            #for k in range(n+1):
            for k in range(self.N - n - 1, self.N):
                #print("k =",k)
                d1, d2 = self.ranklist[k], self.ranklist[k+1]
                V[:,:,k,0:d1,0:d2] = torch.matmul(z, self.TT[k].permute(1,0,2).reshape(self.s, d1*d2)).reshape(b, 1, d1, d2)
                #print(V[:,:,k,0:d1,0:d2].shape)
                f = torch.matmul(f, V[:, :, k,0:d1,0:d2])

            out += f.reshape(b,-1) / factorial(n+1)
            #print("out", out.shape)
        return out + self.b
        

    def forwardOld(self, z, b):
        V = torch.zeros(b, 1, self.N, self.rank, self.rank)
        for k in range(self.N):
            d1, d2 = self.ranklist[k], self.ranklist[k+1]
            # print(self.TT[k].shape)
            # print(self.TT[k].permute(1,0,2).reshape(self.s, d1*d2).shape)
            V[:,:,k,0:d1,0:d2] = torch.matmul(z, self.TT[k].permute(1,0,2).reshape(self.s, d1*d2)).reshape(b, 1, d1, d2)

        f = self.TT[0][0]
        for k in range(self.N):
            d1, d2 = self.ranklist[k], self.ranklist[k+1]
            f = torch.matmul(f, V[:,:, k,0:d1,0:d2])
        return f.reshape(-1)


class Generator(torch.nn.Module):
    def __init__(self, layer, N, rank, imwidth, imheight, scalefactor, layeroptions, generatoroptions):
        super(Generator, self).__init__()
        # Channels: here we are working with greyscale images
        self.c = 1
        self.imwidth, self.imheight = imwidth, imheight
        self.s = imwidth*imheight
        self.PolyLayer = layer(N, rank, imwidth, imheight, layeroptions)
        self.BN = torch.nn.BatchNorm2d(num_features=1)
        self.upsample = torch.nn.Upsample(scale_factor=scalefactor, mode='bilinear', align_corners=False)
        # Number of workers per computation
        self.parallel = generatoroptions['parallel']
        self.workers = generatoroptions['workers']

    def generatorInSequence(self):
        for batch in range(self.batch_size):
            self.x[batch, self.c-1] = self.PolyLayer(self.x[batch, self.c-1])
        return

    def batchesInParallel(self, start, stop, queueindex):
        # self.c-1 because for one channel we want to access index 0
        for i in range(start, stop):
            self.x[i, self.c-1] = self.PolyLayer(self.x[i, self.c-1], queueindex)
        return

    def generatorInParallel(self):
        # Start n number of tasks per number of self.workers
        n_tasks = int(self.batch_size/self.workers)
        schedule = torch.ones(self.workers)*n_tasks
        # Don't forget the remaining ones!
        schedule[0] += self.batch_size - n_tasks*self.workers
        if schedule.sum() != self.batch_size:
            raise ValueError
        # Add the appropriate number of queues to the Layer, don't forget there is a standard Queue already available:
        queuelength = len(self.PolyLayer.QueueList)
        if queuelength != self.workers:
            for queueindex in range(self.workers - 1):
                self.PolyLayer.QueueList.append(queue.Queue())
        # Start threads
        threads = []
        cumul = 0
        for queueindex, group in enumerate(schedule):
            # Perform parallel computation of the rank summation: tremendous speedup.
            start = cumul
            cumul += int(group.item())
            stop = cumul
            currentThread = threading.Thread(target=self.batchesInParallel, args=(start, stop, queueindex))
            currentThread.start()
            threads.append(currentThread)
        # Wait for threads to end:
        for t in threads:
            t.join()
        return

    def forward(self, x):
        # Register dimensions:
        xshape = x.shape
        if len(xshape) == 2:
            self.batch_size = 1
        else:
            self.batch_size = xshape[0]

        # Register x as attribute for parallel access, and clone because dataset would be overwritten
        self.x = self.BN(x.clone())
        self.x = self.upsample(self.x)
        self.x = self.x.reshape(self.batch_size, self.c, self.s)

        if self.parallel:
            self.generatorInParallel()
        else:
            self.generatorInSequence()

        self.x = self.x.reshape(self.batch_size, self.c, self.imwidth, self.imheight)
        return self.x



class Generator_seqOld(torch.nn.Module):
    def __init__(self, layer, N, rank, imwidth, imheight, scalefactor, layeroptions, generatoroptions):
        super(Generator_seq, self).__init__()
        # Channels: here we are working with greyscale images
        self.c = 1
        self.imwidth, self.imheight = imwidth, imheight
        self.s = imwidth*imheight
        self.PolyLayer = layer(N, rank, imwidth, imheight, layeroptions)
        self.BN = torch.nn.BatchNorm2d(num_features=1)
        self.upsample = torch.nn.Upsample(scale_factor=scalefactor, mode='bilinear', align_corners=False)

    def forward(self, x):
        # Register dimensions:
        xshape = x.shape
        if len(xshape) == 2:
            self.batch_size = 1
        else:
            self.batch_size = xshape[0]

        # Register x as attribute for parallel access, and clone because dataset would be overwritten
        self.x = self.BN(x.float())
        #self.x = x.float()#.clone()
        self.x = self.upsample(self.x)
        self.x = self.x.reshape(self.batch_size, self.c, self.s)
        self.x = self.PolyLayer(self.x, self.batch_size)

        self.x = self.x.reshape(self.batch_size, self.c, self.imwidth, self.imheight)
        return self.x

    
class Generator_seq(torch.nn.Module):
    def __init__(self, layer, N, rank, imwidth, imheight, scalefactor, layeroptions, generatoroptions):
        super(Generator_seq, self).__init__()
        # Channels: here we are working with greyscale images
        self.c = 1
        self.imwidth, self.imheight = imwidth, imheight
        self.s = imwidth*imheight
        self.PolyLayer = layer(N, rank, imwidth, imheight, layeroptions)
        self.BN = torch.nn.BatchNorm1d(num_features=self.s)
        self.upsample = torch.nn.Upsample(scale_factor=scalefactor, mode='bilinear', align_corners=False)

    def forward(self, x):
        # Register dimensions:
        xshape = x.shape
        if len(xshape) == 2:
            self.batch_size = 1
        else:
            self.batch_size = xshape[0]

        # Register x as attribute for parallel access, and clone because dataset would be overwritten
        self.x = self.upsample(x.float())
        self.x = self.x.reshape(self.batch_size, self.s)
        self.x = self.BN(self.x).unsqueeze(1)
        self.x = self.PolyLayer(self.x, self.batch_size)
        self.x = self.x.reshape(self.batch_size, self.c, self.imwidth, self.imheight)
        return self.x
