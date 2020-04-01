#!/usr/bin/env python
# coding: utf-8

# # PolyGAN layers
# This notebook contains the different polynomial approximation layers for us in the generator structure

# In[2]:


import torch
import skimage


# In[21]:


# Save the desired order and rank of the following algorithms here:
N = 3
rank = 2

# Save a standard set of inputs
imwidth, imheight = 512, 512
high_res_sample = torch.rand(imwidth,imwidth).float()
low_res_sample  = torch.rand(int(imwidth/4), int(imwidth/4)).float()


# ## PolyGAN CP decomposition
# Here we implement the basic CP for the polyGAN polynomial approximation, Eq (4)

# In[4]:


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
        
        # bias and weights
        b = torch.empty(self.s)
        torch.nn.init.xavier_normal_(b)
        self.b = torch.nn.Parameter(b)
        
        self.W = torch.nn.ParameterList()
        
    
    def forward(self, z):
        # Compute the forward pass: the nmode multiplications f = self.b + self.b + self.W[0]*z + z.T*...
       
        return z
    


# ## PolyGAN TT decomposition
# Here we implement the basic TT for the polyGAN polynomial approximation, Eq (4)

# In[5]:


class PolyGAN_TT_Layer(torch.nn.Module):
    def __init__(self, N, rank, imwidth, imheight, verbose = 0):
        super(PolyGAN_TT_Layer, self).__init__()
        
        # N = order of the polynomial = order of the tensor A:
        # A is of dimension (s, s, ..., s) = (N x s)
        # rank = rank used for the tensor cores
        # size = length of the flattened input
        self.N = N
        self.rank = rank
        self.s = imwidth * imheight
        
        # bias and weights
        b = torch.empty(self.s)
        torch.nn.init.xavier_normal_(b)
        self.b = torch.nn.Parameter(b)
        
        self.W = torch.nn.ParameterList()
        
    
    def forward(self, z):
        # Compute the forward pass: the nmode multiplications f = self.b + self.b + self.W[0]*z + z.T*...
       
        return z
    


# ## Fast Tensor-train contraction
# See "Parallelized Tensor Train Learning of Polynomial
# Classifiers". Eq (8), (10), (11) and algorithm 1.

# The result is given by
# \begin{align*}
# \mathcal{G_1} (i_1) \cdot \prod_{k=2}^{n+1} \left(\sum_{i_k = 1}^{s} z_{i_k} \cdot \mathcal{G_k}(i_k) \right)
# \end{align*}
# with $\boldsymbol{z} \in \mathcal{R}^{d}$

# In[42]:


import threading

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
            self.TT.append(torch.nn.Parameter(TTcore))
                        
        if self.verbose != 0:
            print("self.ranklist =", self.ranklist)
            print("self.N =", self.N, "+ 1 ?=", len(self.ranklist), "= len(self.ranklist)") 
            print("TT has", len(self.TT), "elements:")
            for i, tt in enumerate(self.TT):
                print("element 1:", tt.shape)
        
    def parallelVecProd(self, index):
        res = 0
        for i in range(self.s):
            res += self.z[i] * self.TT[index][:,i,:]
        self.V[index] = res
        # Problem: why does self.V[index] = self.z @ self.TT[index] give a faster but different result?
        # Tested this with both implementations. 
        # Would we need .permute(1,0,2)
        return
    
    def forward(self, z):
        # Compute the forward pass: the nmode multiplications f = A x1 z x2 z x3 ··· x(N-1) z
        # Follow algorithm 1: allocate space and compute each V^(k), possible in parallel with threads.
        self.V = [None] * self.N
        self.z = z
        threads = []

        # Perform parallel computation of the products: tremendous speedup. V[0] will not be used.
        if self.verbose != 0:
            print("Threads are computing..")
        for k in range(1, self.N):
            process = threading.Thread(target=self.parallelVecProd, args=(k,))
            process.start()
            threads.append(process)
            
        # Start the whole product chain now, so that we have [(1),s,r] x [r, r] x ... x [r, (1)] = s
        f = self.TT[0][0,:,:] 
        for k in range(1, self.N):
            threads[k-1].join()
            f @= self.V[k]
            print(self.V[k].shape)
        # Now we have a vector f of size s
        print("f has shape", f.shape)
        return f.reshape(-1)
    


# ## Plug and play
# Test the different layers in a standard net

# In[41]:


# define testnetwork
class Generator(torch.nn.Module):
    def __init__(self, layer, N, rank, imwidth, imheight):
        super(Generator, self).__init__()
        
        self.imwidth, self.imheight = imwidth, imheight
        self.s = imwidth*imheight
        self.PolyLayer = layer(N, rank, imwidth, imheight, 1)
    
    def forward(self, x): 
        # UserWarning: Bi-quadratic interpolation behavior has changed due to a bug in the implementation of scikit-image
        # Perhaps another bilinear interpolation method?
        x = skimage.transform.resize(x, (self.imwidth, self.imheight), order=1, anti_aliasing=True)
        x = torch.tensor(x).float() # make tensor, no need for the very high precision
        x = x.reshape(self.s) # flatten to the 1D equivalent vector
        
        x = self.PolyLayer(x)
        x = x.reshape(self.imwidth, self.imheight)
        return x

net = Generator(FTT_Layer, N, rank, imwidth, imheight)

output = net(low_res_sample)
print("\noutput has shape", output.shape)


# In[7]:


# Time the learning process:
import time
from tqdm import tqdm

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
loops = 10

net.train()

pred_timer = []
crit_timer = []
optm_timer = []
loss_timer = []
step_timer = []

start = time.time()

for t in tqdm(range(loops)):
    # Forward pass: Compute predicted y by passing x to the model
    pred_start = time.time()
    pred = net(low_res_sample)
    pred_timer.append(time.time() - pred_start)
    
    crit_start = time.time()
    loss = criterion(pred, high_res_sample)
    crit_timer.append(time.time() - crit_start)
    
    optm_start = time.time()
    optimizer.zero_grad()
    optm_timer.append(time.time() - optm_start)
    
    loss_start = time.time()
    loss.backward()
    loss_timer.append(time.time() - loss_start)
    
    step_start = time.time()
    optimizer.step()
    step_timer.append(time.time() - step_start)

total_elapsed = time.time() - start
elapsed_per_loop = float(total_elapsed/loops)


# In[8]:


mean = lambda x : sum(x)/len(x)

print("Total elapsed time =", total_elapsed, "s, with", elapsed_per_loop, "s per loop")
print("pred_timer =", mean(pred_timer), "s on average")
print("crit_timer =", mean(crit_timer), "s on average")
print("optm_timer =", mean(optm_timer), "s on average")
print("loss_timer =", mean(loss_timer), "s on average")
print("step_timer =", mean(step_timer), "s on average")

