STDC version 1,
Released by author Yi-Lei Chen,

2012/12/28

#####################################################################################################################

<Description of the STDC function>


[V,Core,info,X,itr,t] = STDC(X,mark,para_ST,mode,Xg)

Input:
   -- X:       an input N-th order tensor object

   -- mark:    boolean index of missing entries (1 indicates missing; 0 otherwise)

   -- para_ST: a structure component including parameters used for the proposed STDC algorithm

   -- mode:    determining whether the N-th submanifold is computed (set as 0) or not (set as 1)
               (because V_N is usually ignored in multilinear model analysis)

   -- Xg:      the ground truth of X; if not available, just using X instead



Parameters of para_ST:

   -- maxitr:     maximum iteration (default:50)

   -- ita_rate:   increased penalty of equality constraint (default:1.1)

   -- tau:        the 1st threshold in shrinkage operation (default:0.1)

   -- kappa:      the weight of MGE (default:10)

   -- omega:      the weight of Z-regularization (default:0.001)

   -- print_mode: determining the algorithm message printed iteratively (1) or finally (0) (default:0)

   -- Rate:       an Mx1 vector, which indicates the downsampling rate of M multilinear graphs

   -- VSet:       an Mx1 cell-array; 
                  each element is an Nx1 boolean vector, which indicates the L-dependent subset of {V_1,...,V_N}

   -- H:          an MxK cell-array (K is the maximum of # submanifolds among M subsets);
                  each element denotes a tensor, whose unfolding matrix is the cholskey decomposition of the 
                  corresponding Laplacian matrix (H'H = L);
                  if the m-th graph is not downsampled, users only have to specify H{m,1}; otherwise,
                  user have to specify H{m,1} to H{m,k} for the k (k<=K) submanifolds in the m-th subset


(Note that, if no factor prior is included, users need not to specify kappa, Rate, VSet, and H.)


#####################################################################################################################

<Execution of STDC>

Just running the following two files in Matlab platform:

test_synthetic.m (for sec. 4.1)
* please download the input file "ex1.mat" from our project website.
(http://mp.cs.nthu.edu.tw/project_STDC/)

test_image.m     (for sec. 4.3)

