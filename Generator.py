import torch
import threading
import queue

import math

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
        # Necessary for threads:
        self.parallel = parallel
        if randnormweights:
            self.initweights = torch.nn.init.xavier_normal_
        else:
            self.initweights = torch.nn.init.xavier_uniform_
        self.QueueList = [queue.Queue()]
        self.Lock = threading.Lock()


class PolyganCPlayer(PolyLayer):
    def __init__(self, N, rank, imwidth, imheight, layeroptions):
        PolyLayer.__init__(self, N, rank, imwidth*imheight, randnormweights=layeroptions['randnormweights'], parallel=layeroptions['parallel'])

        # Initialize the bias
        b = torch.empty(self.s,1)
        self.initweights(b)
        if layeroptions['normalize']:
            b /= self.s
        self.b = torch.nn.Parameter(b.reshape(self.s))
        # Initialize the weights
        self.W = torch.nn.ParameterList()
        self.shapelist = []
        # 0th order: bias [s], 1st order: weight [s, s], 2nd order: weight [s, s, s], ...
        for n in range(1, N + 1):
            tshape = [self.s]*(n + 1)
            self.shapelist.append(tshape)
            for o in range(n + 1):
                factor_matrix = torch.zeros((self.s, self.rank))
                self.initweights(factor_matrix)
                if layeroptions['normalize']:
                    factor_matrix /= self.s
                self.W.append(torch.nn.Parameter(factor_matrix))

    def forwardInSequence(self, z, queue):
        # Simple and straightforward: see notes on github
        Rsums = queue.get()
        for n in range(self.N):
            partialsum = 0
            for r in range(self.rank):
                f = 1
                for k in range(1, n):
                    f *= torch.dot(z, self.W[n][:,r])
                partialsum += self.W[0][:,r] * f
            Rsums[n] = partialsum
        queue.put(Rsums)
        return

    def forwardInSequenceTest(self, z, queue):
        # Simple and straightforward: see notes on github
        Rsums = queue.get()
        for n in range(self.N):
            partialsum = 0
            f = torch.ones(self.rank)
            for k in range(1, n):
                f *= torch.dot(z, self.W[k])
            print("f = " + str(f.shape) + ", W = " + str(self.W[0].shape))
            partialsum += f @ self.W[0] 
            print("partialsum = " + str(partialsum.shape))
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
            currentThread.setDaemon(True)
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
            TTcore = torch.empty(self.ranklist[n], self.s, self.ranklist[n+1])
            self.initweights(TTcore)
            if layeroptions['normalize']:
                TTcore /= math.pow(self.s, 3)
                #TTcore /= math.pow(1.e25, 1/10)
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
            currentThread.setDaemon(True)
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
        self.Lock = threading.Lock()

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
            currentThread.setDaemon(True)
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
