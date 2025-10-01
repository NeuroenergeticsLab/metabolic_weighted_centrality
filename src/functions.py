#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from sklearn.decomposition import PCA
import scipy as sp
import numpy as np
import pandas as pd
import nibabel as nib
import warnings   
import os
import pickle
import networkx as nx
from brainsmash.mapgen.base import Base
from brainsmash.mapgen.eval import base_fit
from brainsmash.mapgen.stats import nonparp, pairwise_r
import plotly.graph_objects as go
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from matplotlib.cbook import boxplot_stats
import igraph as ig
from scipy.stats import norm

np.float = float
np.int = int
# In[6]:


def pcor (x,y):
    from scipy.stats import pearsonr
    non_nan_indices = ~np.isnan(x) & ~np.isnan(y)
    xx = x[non_nan_indices]
    yy = y[non_nan_indices]
    corr = pearsonr(xx,yy)

    # bad = ~np.logical_or(np.isnan(x), np.isnan(y))
    #xx = np.compress(bad, x)  
    #yy = np.compress(bad, y) 

    corr, pvalue = pearsonr(xx, yy)  # Unpack both values here
    return corr, pvalue





def threshold_proportional(W, p, copy=True):
    '''
    This function "thresholds" the connectivity matrix by preserving a
    proportion p (0<p<1) of the strongest weights. All other weights, and
    all weights on the main diagonal (self-self connections) are set to 0.
    If copy is not set, this function will *modify W in place.*
    Inputs: W,      weighted or binary conneccivity matrix
            p,      proportion of weights to preserve
                        range:  p=1 (all weights preserved) to
                                p=0 (nco weights preserved)
            copy,    copy W to avoid side effects, defaults to True
    Output: W,        thresholded connectivity matrix
    Note: The proportion of elements set to 0 is a fraction of all elements in the 
    matrix, whether or not they are already 0. That is, this function has the
    following behavior:
        >> x = np.random.random((10,10))
        >> x_25 = threshold_proportional(x, .25)
        >> np.size(np.where(x_25)) #note this double counts each nonzero element
        46
        >> x_125 = threshold_proportional(x, .125)
        >> np.size(np.where(x_125))
        22
        >> x_test = threshold_proportional(x_25, .5)
        >> np.size(np.where(x_test))
        46
    That is, the 50% thresholding of x_25 does nothing because >=50% of the elements
    in x_25 are aleady <=0. This behavior is the same as in BCT. Be careful with matrices that are both signed and sparse.
    '''
    if p > 1 or p < 0:
        raise BCTParamError('Threshold must be in range [0,1]')
    if copy:
        W = W.copy()
    n = len(W)                        # number of nodes
    np.fill_diagonal(W, 0)            # clear diagonal

    if np.all(W == W.T):                # if symmetric matrix
        W[np.tril_indices(n)] = 0        # ensure symmetry is preserved
        ud = 2                        # halve number of removed links
    else:
        ud = 1

    ind = np.where(W)                    # find all links

    I = np.argsort(W[ind])[::-1]        # sort indices by magnitude

    # number of links to be preserved
    en = round((n * n - n) * p / ud)

    W[(ind[0][I][en:], ind[1][I][en:])] = 0    # apply threshold

    if ud == 2:                          # if symmetric matrix
        W[:, :] = W + W.T                        # reconstruct symmetry

    return W



def IC_calculation(FCmat, pet, SCmat, thr_sc, thr_FC, sub_size, scmask, nrois_rem, net_label):





    data_single_sub = {
        'ICallsub_w': np.zeros((nrois_rem, sub_size)),
        'ICallsub_b': np.zeros((nrois_rem, sub_size)),
        'degallsub_w': np.zeros((nrois_rem, sub_size)),
        'degallsub_b': np.zeros((nrois_rem, sub_size)),
        'AvgMIallsub': np.zeros((nrois_rem, sub_size)),
        'eig_cenallsub': np.zeros((nrois_rem, sub_size)),
        'btw_cenallsub': np.zeros((nrois_rem, sub_size)),
    }

    SCmat_th = np.zeros((SCmat.shape[0], SCmat.shape[1], sub_size))

    for i in range(sub_size):
        SCmat_th[:, :, i] = threshold_proportional(SCmat[:, :, i], thr_sc)

    SC_mask = SCmat_th.copy()
    SC_mask[SC_mask > 0] = 1



    pet_rem = pet.copy()



    for j in range(sub_size):

        FCconn = FCmat[:, :, j]
        SCconn = SC_mask[:,:,j]



        FCconn_th = threshold_proportional(FCconn, thr_FC)

        # fc_neighbor_mask = connectivity_matrix_allsub[:, :, j]
        # FCconn_th = FCconn_th * fc_neighbor_mask

        if scmask == 1:
            FCmat_SC = np.multiply(FCconn_th , SCconn)

        else:
            FCmat_SC = FCconn_th
        CMR2 = pet_rem[:, j]

        #FCconn_th = FCconn
        deg_w = np.nansum(FCmat_SC, axis=1)
        FCconn_th_b = FCmat_SC.copy()
        FCconn_th_b[FCconn_th_b > 0] = 1
        deg_b = np.nansum(FCconn_th_b, axis=1)
        E_MAT = np.tile(CMR2, (nrois_rem, 1)) #########


        GFC = nx.Graph(FCmat_SC)

        #eigenvector_centrality = nx.eigenvector_centrality(GFC)
        #eig_cen = list(eigenvector_centrality.values())

        betweenness_centrality = nx.betweenness_centrality(GFC , normalized=True, weight = 'weight')
        btw_cen = list(betweenness_centrality.values())

        #FCconn_thnor = np.zeros((nrois_rem, nrois_rem))
        #FCconn_thnor_b = np.zeros((nrois_rem, nrois_rem))

       # for r in range(nrois_rem):
          #  for s in range(nrois_rem):
              #  FCconn_thnor[r, s] = np.divide(FCconn_th[r, s] , np.nansum(FCconn_th[r, :]))
               # FCconn_thnor_b[r, s] = np.divide(FCconn_th_b[r, s]  , np.nansum(FCconn_th_b[r, :]))

        row_sums = np.nansum(FCmat_SC, axis=1)  # Compute the sum of each row
        FCconn_thnor = np.divide(FCmat_SC , row_sums[:, np.newaxis], out=np.zeros_like(FCmat_SC ), where=row_sums[:, np.newaxis] != 0)

        row_sums_b = np.nansum(FCconn_th_b, axis=1)  # Compute the sum of each row
        FCconn_thnor_b = np.divide(FCconn_th_b, row_sums_b[:, np.newaxis], out=np.zeros_like(FCconn_th_b), where=row_sums[:, np.newaxis] != 0)

        IC_w = np.nansum(np.multiply(FCconn_thnor , E_MAT), axis=1)
        IC_b = np.nansum(np.multiply(FCconn_thnor_b ,E_MAT), axis=1)
        AvgMI = np.divide(deg_w , deg_b)

        data_single_sub['ICallsub_w'][:, j] = IC_w
        data_single_sub['ICallsub_b'][:, j] = IC_b
        data_single_sub['degallsub_w'][:, j] = deg_w
        data_single_sub['degallsub_b'][:, j] = deg_b
        data_single_sub['AvgMIallsub'][:, j] = AvgMI
        #data_single_sub['eig_cenallsub'][:, j] = eig_cen
        data_single_sub['btw_cenallsub'][:, j] = btw_cen


    data_single_sub['ICallsub_w'][data_single_sub['ICallsub_w'] == 0] = np.nan
    data_single_sub['ICallsub_b'][data_single_sub['ICallsub_b'] == 0] = np.nan
    data_single_sub['degallsub_w'][data_single_sub['degallsub_w'] == 0] = np.nan
    data_single_sub['degallsub_b'][data_single_sub['degallsub_b'] == 0] = np.nan
    data_single_sub['AvgMIallsub'][data_single_sub['AvgMIallsub']== 0] = np.nan
    #data_single_sub['eig_cenallsub'][data_single_sub['eig_cenallsub'] == 0] = np.nan
    data_single_sub['btw_cenallsub'][data_single_sub['btw_cenallsub'] == 0] = np.nan


    degallsub_w_avg = np.nanmean(data_single_sub['degallsub_w'], axis=1)
    degallsub_b_avg = np.nanmean(data_single_sub['degallsub_b'], axis=1)
    pet_avg = np.nanmean(pet_rem, axis=1)
    ICallsub_w_avg = np.nanmean(data_single_sub['ICallsub_w'], axis=1)
    ICallsub_b_avg = np.nanmean(data_single_sub['ICallsub_b'], axis=1)
    AvgMIallsub_avg = np.nanmean(data_single_sub['AvgMIallsub'], axis=1)
    #eig_cenallsub_avg = np.nanmean(data_single_sub['eig_cenallsub'], axis=1)
    btw_cenallsub_avg = np.nanmean(data_single_sub['btw_cenallsub'], axis=1)


    data_avg = pd.DataFrame({'degallsub_w_avg' : degallsub_w_avg ,'degallsub_b_avg' : degallsub_b_avg,
                       'pet_avg' : pet_avg , 'ICallsub_w_avg' : ICallsub_w_avg , 
                         'ICallsub_b_avg' : ICallsub_b_avg, 'AvgMIallsub_avg': AvgMIallsub_avg,  'btw_avg': btw_cenallsub_avg})#'eig_avg': eig_cenallsub_avg,


    return data_avg, data_single_sub
# def pcor (x,y):

#     bad = ~np.logical_or(np.isnan(x), np.isnan(y))
#     xx = np.compress(bad, x)  
#     yy = np.compress(bad, y)  
#     corr = pearsonr(xx,yy)
#     return corr




def Spatial_AC(DATA_avg, ic, LIMB,niter):
    from scipy import stats

    from brainsmash.mapgen.stats import pearsonr
    if LIMB == "without":
        dist_file = "../data/external/LeftParcelGeodesicDistmat_wolimb.txt"
    else:
        dist_file = "../data/external/LeftParcelGeodesicDistmat.txt"


    ## ic can be IC or deg_w or deg_b: 



    ## Contain the limbic network regions for SAC: 

    # load parcellated neuroimaging maps






    if ic == "IC":
        half_size = len(DATA_avg.ICallsub_w_avg.values) // 2  
        pet = DATA_avg.pet_avg.values[0:half_size]
        IC = DATA_avg.ICallsub_w_avg.values[0:half_size]


    elif ic == "deg_w":

        half_size = len(DATA_avg.ICallsub_w_avg.values) // 2 
        IC = DATA_avg.degallsub_w_avg.values[0:half_size]
        pet = DATA_avg.pet_avg.values[0:half_size]

    elif ic == "PC":

        half_size = len(DATA_avg.ICallsub_w_avg.values) // 2 
        IC = DATA_avg.PCallsub_w_avg.values[0:half_size]
        pet = DATA_avg.pet_avg.values[0:half_size]

    elif ic == "deg_b":

        half_size = len(DATA_avg.ICallsub_w_avg.values) // 2 
        IC = DATA_avg.degallsub_b_avg.values[0:half_size]
        pet = DATA_avg.pet_avg.values[0:half_size]
    else:
        x_ser = DATA_avg['x']
        y_ser = DATA_avg['y']
        half_size = len(x_ser) // 2        
        IC  = x_ser.to_numpy()[:half_size]
        pet = y_ser.to_numpy()[:half_size]





    # instantiate class and generate 1000 surrogates
    gen = Base(pet, dist_file) 
    surrogate_maps = gen(n = niter)


    surrogate_brainmap_corrs = pearsonr(IC, surrogate_maps).flatten()
    surrogate_pairwise_corrs = pairwise_r(surrogate_maps, flatten=True)

    naive_surrogates = np.array([np.random.permutation(pet) for _ in range(niter)])
    naive_brainmap_corrs = pearsonr(IC, naive_surrogates).flatten()
    naive_pairwise_corrs = pairwise_r(naive_surrogates, flatten=True)

    sac = '#377eb8'  # autocorr-preserving
    rc = '#e41a1c'  # randomly shuffled
    bins = np.linspace(-1, 1, 51)  # correlation b

    # this is the empirical statistic we're creating a null distribution for
    test_stat = stats.pearsonr(pet, IC)[0]

    base_fit(
        x=pet,
        D=dist_file,
        nsurr= niter,
        nh=25,  # these are default kwargs, but shown here for demonstration
        deltas=np.arange(0.1, 1, 0.1),
        pv=25)  # kwargs are passed to brainsmash.mapgen.base.Base

    spatially_naive_p_value = nonparp(test_stat, naive_brainmap_corrs)
    sa_corrected_p_value = nonparp(test_stat, surrogate_brainmap_corrs)



    test_stat = stats.pearsonr(pet, IC)[0]
    print("Pearson correlation:", test_stat)



    print("Spatially naive p-value: {:.2e}".format(spatially_naive_p_value))
    print("SA-corrected p-value: {:.2e}".format(sa_corrected_p_value))


    return test_stat ,surrogate_brainmap_corrs, sa_corrected_p_value, spatially_naive_p_value


# In[2]:


"""
Gaussian copula mutual information estimation
"""


#warnings.filterwarnings('ignore')


__version__ = '0.3'

def ctransform(x):
    """Copula transformation (empirical CDF)
    cx = ctransform(x) returns the empirical CDF value along the first
    axis of x. Data is ranked and scaled within [0 1] (open interval).
    """

    xi = np.argsort(np.atleast_2d(x))
    xr = np.argsort(xi)
    cx = (xr+1).astype(float) / (xr.shape[-1]+1)
    return cx


def copnorm(x):
    """Copula normalization

    cx = copnorm(x) returns standard normal samples with the same empirical
    CDF value as the input. Operates along the last axis.
    """
    cx = sp.stats.norm.ppf(ctransform(x))
    #cx = sp.special.ndtri(ctransform(x))
    return cx





def mi_gg(x, y, biascorrect=True, demeaned=False):
    """Mutual information (MI) between two Gaussian variables in bits

    I = mi_gg(x,y) returns the MI between two (possibly multidimensional)
    Gassian variables, x and y, with bias correction.
    If x and/or y are multivariate columns must correspond to samples, rows
    to dimensions/variables. (Samples last axis) 

    biascorrect : true / false option (default true) which specifies whether
    bias correction should be applied to the esimtated MI.
    demeaned : false / true option (default false) which specifies whether th
    input data already has zero mean (true if it has been copula-normalized)
    """

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    if x.ndim > 2 or y.ndim > 2:
        raise ValueError("x and y must be at most 2d")
    Ntrl = x.shape[1]
    Nvarx = x.shape[0]
    Nvary = y.shape[0]
    Nvarxy = Nvarx+Nvary

    if y.shape[1] != Ntrl:
        raise ValueError("number of trials do not match")

    # joint variable
    xy = np.vstack((x,y))
    if not demeaned:
        xy = xy - xy.mean(axis=1)[:,np.newaxis]
    Cxy = np.dot(xy,xy.T) / float(Ntrl - 1)
    # submatrices of joint covariance
    Cx = Cxy[:Nvarx,:Nvarx]
    Cy = Cxy[Nvarx:,Nvarx:]

    chCxy = np.linalg.cholesky(Cxy)
    chCx = np.linalg.cholesky(Cx)
    chCy = np.linalg.cholesky(Cy)

    # entropies in nats
    # normalizations cancel for mutual information
    HX = np.sum(np.log(np.diagonal(chCx))) # + 0.5*Nvarx*(np.log(2*np.pi)+1.0)
    HY = np.sum(np.log(np.diagonal(chCy))) # + 0.5*Nvary*(np.log(2*np.pi)+1.0)
    HXY = np.sum(np.log(np.diagonal(chCxy))) # + 0.5*Nvarxy*(np.log(2*np.pi)+1.0)

    ln2 = np.log(2)
    if biascorrect:
        psiterms = sp.special.psi((Ntrl - np.arange(1,Nvarxy+1)).astype(np.float)/2.0) / 2.0
        dterm = (ln2 - np.log(Ntrl-1.0)) / 2.0
        HX = HX - Nvarx*dterm - psiterms[:Nvarx].sum()
        HY = HY - Nvary*dterm - psiterms[:Nvary].sum()
        HXY = HXY - Nvarxy*dterm - psiterms[:Nvarxy].sum()

    # MI in bits
    I = (HX + HY - HXY) / ln2
    return I



def ConnectivityMatrices(data, nvox, nT, Parc1, GM, VOX, num_pcs, ABS, limb_ind):

    if VOX == 1:

        rem_ind = limb_ind
        Parcfmri_gm = np.multiply(Parc1 , GM)
        #Parcfmri_gm = Parc1 
        Parcfmri_gm_re = np.reshape(Parcfmri_gm, [nvox, 1],order='F')
        parcindfmri_gm = np.unique(Parcfmri_gm_re)
        parcindfmri_gm = parcindfmri_gm[1:]
        nrois_gm = len(np.unique(Parcfmri_gm)) - 1

        Parc1_re = np.reshape(Parc1, [nvox, 1],order='F')
        nrois = len(np.unique(Parc1_re)) - 1
        nrois_rem = nrois - len(rem_ind)
        rois_remain = np.arange(1, nrois+1 )
        rois_remain[limb_ind ] = 0  
        rois_remain = rois_remain[rois_remain != 0] 

        data_re = np.reshape(data, [nvox, nT] ,order='F')
        Parcindfmri = np.unique(Parc1)
        Parcindfmri = Parcindfmri[1:]
        DATA_avg = np.zeros([nT, nrois])
        num_voxels = np.zeros((nrois))
        DATA = {}



        for i in range(nrois):

            if np.any(Parcfmri_gm_re == Parcindfmri[i]):

                ind = np.where(Parcfmri_gm_re == Parcindfmri[i])[0]

                dd = data_re[ind, :]

                # indremove = np.where(np.all(dd == 0, axis=1))[0]
                # dd = np.delete(dd, indremove, axis=0)

                # if indremove.size == ind.size:
                #     zero_rois[i] = 1

                DATA[i] = dd
                num_voxels[i] = dd.shape[0]
                # edata_median[i, 0] = np.median(edata_re[ind])
                DATA_avg[:, i] = np.mean(dd, axis=0)
            else:
                DATA_avg[:, i] = np.nan 
                DATA[i] = np.zeros((10,1)) 


    else:
        DATA_avg = data
        #nrois = nvox
       # nrois_rem = nrois - len(rem_ind)



    Pearconn = np.corrcoef(DATA_avg, rowvar=False)
    Pearconn = np.delete(Pearconn , rem_ind , axis =0)
    Pearconn = np.delete(Pearconn , rem_ind , axis =1)

    # MVMIconn = np.zeros((nrois_rem, nrois_rem))
    MIconn = np.zeros((nrois_rem, nrois_rem))
    cDATA_avg = np.transpose(copnorm(np.transpose(DATA_avg)))
    np.delete(cDATA_avg, limb_ind)
    zero_rois = np.zeros(nrois_rem)
    Cdata_abs = {}
    Cdata = {}
    for i in range(nrois_rem):

        X = DATA[ rois_remain[i]-1]
        ind = np.where(np.sum(X, axis=0) == 0)[0]
        if np.any(X) == True:
            X = np.delete(X, ind, axis=0)
        else:
            zero_rois[i] = 1



        quan = np.quantile(num_voxels, [0.25, 0.5, 0.75, 1])
        if num_voxels[rois_remain[i]-1] > quan[2]:
            num_pcs = 5
        elif num_voxels[rois_remain[i]-1] > quan[1]:
            num_pcs = 4
        elif num_voxels[rois_remain[i]-1] > quan[0]:
            num_pcs = 3
        else:
            num_pcs = 2


        pca = PCA()
        X_score = pca.fit_transform(np.transpose(X))
        CC = copnorm(X_score[:, 0:min(num_pcs, X_score.shape[1])].T)
        Cdata_abs[i] = np.hstack((CC, copnorm(np.abs(CC))))
        Cdata[i] = CC

    for i in range(nrois_rem):

        if zero_rois[i]==1:
            MIconn[i,:]=np.nan
           # MVMIconn[i,:]=np.nan
            continue

        CX = Cdata[i]
        CX2 = Cdata_abs[i]

        for j in range(i+1, nrois_rem):

            if zero_rois[j]==1:
                MIconn[:,j]=np.nan
                #MVMIconn[:,j]=np.nan
                continue

            MIconn[j,i]= mi_gg(cDATA_avg[:,rois_remain[j]-1],cDATA_avg[:,rois_remain[i]-1])  

            CY = Cdata[j]     
            CY2 = Cdata_abs[j]

           # if ABS == 1:
               # num = min(CX2.shape[1], CY2.shape[1])
               # MVMIconn[i,j] = mi_gg(CX2[:, :num], CY2[:, :num])
            #else:
               # MVMIconn[i,j] = mi_gg(CX, CY)


    MIconn = MIconn + MIconn.T




    #MVMIconn = MVMIconn + MVMIconn.T

    return num_voxels  , MIconn , Pearconn


def IC_model(X):
    E = X[:len(X)//2]
    alpha = X[len(X)//2:]
    IC = np.nansum(np.multiply(E,alpha))
    return IC









# In[3]:


def Functional_conn(DIR, session, sub, limb_ind, nrois):
    
    data_input = {}
    
    warnings.filterwarnings('ignore')

    #sub = [3,7,12,14,17,20,23,25,26,28,29,30,31,32,33,35,36,37,38]; 
    #sub = [3,7,12,14,17,20,23,25,26,28,29,30,31,33,35,36,37,38]; 

    #subcor = 15 #***with subcortical******
    subcor = 0
    remain_rois = range(0 , nrois)
    remain_rois = np.delete(remain_rois , limb_ind)
    nrois_rem = nrois - len(limb_ind)
    mask_rois = np.zeros((nrois, len(sub)))
    rois_remain = np.zeros((nrois_rem, len(sub)))

    data_input['num_voxels'] = np.zeros((nrois, len(sub) ))
    data_input['edata_medianallsub'] = np.zeros((nrois_rem, len(sub) ))
    data_input['MIconn_allsub'] = np.zeros((nrois_rem, nrois_rem, len(sub)))
    data_input['SCconn_allsub'] = np.zeros((nrois_rem, nrois_rem, len(sub) ))
    data_input['Pearconn_allsub'] = np.zeros((nrois_rem, nrois_rem, len(sub) ))


    for b ,subject in enumerate(sub):
        
        #loading the GM and parcellation in pet and fmri spaces

        DIR1 = f'{DIR}sub-{subject:03}/'
        Parcfmri =  nib.load(f'{DIR1}MMP_in_func3mm.nii.gz').get_fdata()
        #Parcfmri_subcor =  40 * nib.load(DIR1 + 'mmp_subcortical_in_func3mm.nii.gz').get_fdata()
        #Parcfmri_subcor_mask =  nib.load(DIR1 + 'mmp_subcortical_in_func3mm_mask.nii.gz').get_fdata()


        Parcpet  =  nib.load(DIR1 + 'MMP_in_pet3mm.nii.gz').get_fdata()
        #Parcpet_subcor  =  40 * nib.load(DIR1 + 'mmp_subcortical_in_pet3mm.nii.gz').get_fdata()
        #Parcpet_subcor_mask =  nib.load(DIR1 + 'mmp_subcortical_in_pet3mm_mask.nii.gz').get_fdata()


        GM_fmri  =  nib.load(DIR1 + "gm_in_func3mm.nii.gz").get_fdata()
        GM_pet   =  nib.load(DIR1 + "gm_in_pet3mm.nii.gz").get_fdata()    


        #for adding subcortical regions uncomment the 4 following lines:

        #indic_func = -Parcfmri_subcor_mask+1
        #indic_pet  = -Parcpet_subcor_mask+1
        #Parcfmri  =   np.multiply(indic_func,Parcfmri) + Parcfmri_subcor #***with subcortical******
        #Parcpet =  np.multiply(indic_pet , Parcpet)  + Parcpet_subcor #***with subcortical******
       # GM_fmri =  GM_fmri + Parcfmri_subcor_mask #***with subcortical******
        #GM_pet  =  GM_pet   + Parcpet_subcor_mask #***with subcortical******


        #loading the fmri and pet data
        BDATA = nib.load(DIR1 + 'func3mm.nii.gz').get_fdata()
        EDATA = nib.load(DIR1 + 'pet3mm.nii.gz').get_fdata()

        v1, v2, v3, v4 = BDATA.shape
        nT = v4
        nvox = v1 * v2 * v3

        v11 , v22 , v33  = EDATA.shape
        nvox2 = v11 * v22 * v33



        Parcpet_gm = np.multiply(Parcpet , GM_pet) 

        Parcpet_re = Parcpet_gm.reshape((nvox2, 1),order='F')
        parcindpet = np.unique(Parcpet)
        parcindpet = parcindpet[1:]
        parcindpet_gm = np.unique(Parcpet_re)
        parcindpet_gm = parcindpet_gm[1:]
        nroispet = len(np.unique(Parcpet_gm)) - 1

        # parcellation on Energy data:
        edata_re = EDATA.reshape((nvox2, 1),order='F')

        for i in range(nrois_rem):

            if parcindpet_gm.__contains__(parcindpet[remain_rois[i]]):

                indpet = np.where(Parcpet_re == parcindpet[remain_rois[i]])[0]
                data_input['edata_medianallsub'][i, b] = np.nanmedian(edata_re[indpet])
            else:

                data_input['edata_medianallsub'][i, b] = np.nan
        [data_input['num_voxels'][:,b],data_input['MIconn_allsub'][:,:,b],data_input['Pearconn_allsub'][:,:,b]] =   ConnectivityMatrices(BDATA, nvox, nT, Parcfmri, GM_fmri, 1, 2, 1, limb_ind) 
        
    data_input['num_voxels'] = np.delete(data_input['num_voxels'] , limb_ind, axis = 0)   

    
    if session == "AUF":
        SCconn_allsub = structural_conn(DIR, sub, nrois, limb_ind)
    else:
        SCconn_allsub = np.ones_like(data_input['MIconn_allsub'])

    data_input['SCconn_allsub'] = SCconn_allsub
 
    return data_input








# ### Structural connectivity

# In[4]:


def structural_conn(DIR, sub, nrois,limb_ind):    


    #sub = [3,7,12,14,17,20,23,25,26,28,29,30,31,33,35,36,37,38];  #***with subcortical******
    # downloading SC matrices:
    SCconn_allsub = np.zeros((nrois, nrois, len(sub)))
    for j in range(len(sub)):
        subject = sub[j]
        DIR1 = os.path.join(DIR, f'sub-{subject:03}')
        #sc_dir = os.path.join(DIR1, 'connectom_glasser_subcortical.csv') #***with subcortical******
        sc_dir = os.path.join(DIR1, 'scmat.csv')
        scmat = pd.read_csv(sc_dir, header=None)
        scmat = scmat.iloc[1:, 1:].to_numpy().astype(float)
        if scmat.shape[0] < 360:
            print(j)
            continue
        SCconn_allsub[:, :, j] = scmat


    SCconn_allsub = np.delete(SCconn_allsub, limb_ind , axis =0)
    SCconn_allsub = np.delete(SCconn_allsub , limb_ind, axis =1)

    return SCconn_allsub


# # Adding mvMI 

# In[4]:




# In[1]:


def steiger_z_test(r12, r13, r23, n):
    """
    Performs Steiger's Z-test to compare two dependent correlation coefficients.

    Parameters:
        r12 (float): Correlation between variable 1 and 2.
        r13 (float): Correlation between variable 1 and 3.
        r23 (float): Correlation between variable 2 and 3.
        n (int): Sample size.

    Returns:
        z (float): Z-score.
        p (float): Two-tailed p-value.
    """
    # Fisher transformation for r12 and r13
    z12 = 0.5 * np.log((1 + r12) / (1 - r12))
    z13 = 0.5 * np.log((1 + r13) / (1 - r13))

    # Compute standard error
    se = np.sqrt((2 * (1 - r23)) / (n - 3))

    # Compute Steiger's Z-score
    z = (z12 - z13) / se

    # Compute two-tailed p-value
    p = 2 * (1 - norm.cdf(abs(z)))

    return z, p


# In[ ]:


def plot_node_surf(data, conn_mat, param, CMAP, edge_thr, limb_ind):
 
    cmat = np.mean(conn_mat, axis=2)


    plotly_cmap = [
        [f, '#{:02x}{:02x}{:02x}'.format(*((rgba[:3] * 255).astype(int)))]
        for f, rgba in zip(np.linspace(0, 1, 256), CMAP(np.linspace(0, 1, 256)))
    ]


    IC = data[param].values.astype(float)
    IC_min = IC.min()
    cap = 34 if param == "ICallsub_w_avg" else IC.max()
    IC_capped = np.clip(IC, IC_min, cap)
    IC_norm = IC
    color_values = IC_capped


    coords = pd.read_csv("/RAID1/jupytertmp/mi/input_data/HCP-MMP1_UniqueRegionList.csv")
    coords = coords[["x-cog", "y-cog", "z-cog"]]
    coords_df = coords.drop(limb_ind, axis=0)
    coords_df.columns = coords_df.columns.str.strip()
    nodes = coords_df[['x-cog', 'y-cog', 'z-cog']].values

   
    G = nx.Graph()
    for idx, node in enumerate(nodes):
        G.add_node(idx, coord=node)
    source, target = np.nonzero(np.triu(cmat) > edge_thr)
    edges = list(zip(source, target))
    G.add_edges_from(edges)

    nodes_x = coords_df['x-cog'].values
    nodes_y = coords_df['y-cog'].values
    nodes_z = coords_df['z-cog'].values

    p85, p90, p95 = np.percentile(IC_norm, [85, 90, 95])
    node_sizes = np.full_like(IC_norm, 8, dtype=np.float32)
    node_sizes[IC_norm >= p85] = 12
    node_sizes[IC_norm >= p90] = 17
    node_sizes[IC_norm >= p95] = 22

    edge_x, edge_y, edge_z = [], [], []
    for s, t in edges:
        edge_x += [nodes_x[s], nodes_x[t], None]
        edge_y += [nodes_y[s], nodes_y[t], None]
        edge_z += [nodes_z[s], nodes_z[t], None]

    fig = go.Figure()

    # --- nodes
    fig.add_trace(go.Scatter3d(
        x=nodes_x, y=nodes_y, z=nodes_z, mode='markers', name='Nodes',
        marker=dict(size=node_sizes, color=color_values, colorscale=plotly_cmap,
                    opacity=0.8, showscale=False),
        showlegend=False, hoverinfo='text'
    ))

    # --- edges
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z, mode='lines', name='Edges',
        line=dict(color='gray'), opacity=0.1, showlegend=False, hoverinfo='none'
    ))


    logical_sizes = [8, 12, 17, 22]
    labels = ["≤ 85th pct", "≥ 85th pct", "≥ 90th pct", "≥ 95th pct"]

   
    leg_min, leg_max = 6.0, 15.0
    s_min, s_max = min(logical_sizes), max(logical_sizes)
    legend_sizes = [leg_min + (s - s_min) * (leg_max - leg_min) / (s_max - s_min)
                    for s in logical_sizes]

    size_levels = np.array([8, 12, 17, 22], dtype=float)   # your original tiers

    # pick a global scale so the figure isn't crowded (tweak 0.55–0.8 to taste)
    size_scale = 0.65
    size_levels_scaled = (size_levels * size_scale).tolist()
    
    # thresholds
    p85, p90, p95 = np.percentile(IC_norm, [85, 90, 95])
    
    # apply to nodes (now nodes and legend will be identical in pixels)
    node_sizes = np.full_like(IC_norm, size_levels_scaled[0], dtype=float)
    node_sizes[IC_norm >= p85] = size_levels_scaled[1]
    node_sizes[IC_norm >= p90] = size_levels_scaled[2]
    node_sizes[IC_norm >= p95] = size_levels_scaled[3]


  
    for label, s in zip(labels, legend_sizes):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=label, hoverinfo="skip", showlegend=True,
            marker=dict(size=s, color="rgba(128,128,128,0.9)", line=dict(color="gray", width=1)),
            legendgroup="size",
        ))

    
    # --- orientation, background, and tight legend placement ---
    fig.update_layout(
        # fill the canvas; leave a slim band for the legend
        scene=dict(
            bgcolor="white",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            domain=dict(y=[0.12, 1.0])   # push the 3D scene up; ~12% space for legend
        ),
        scene_camera=dict(
            eye=dict(x=0.0, y=1.8, z=0.08),  # frontal view; smaller values = more zoom
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
        ),
    
        # legend right below the scene, centered
        legend=dict(
            orientation="h",
            itemsizing="trace",
            x=0.5, xanchor="center",
            y=0.15, yanchor="top"          # sits just under the scene
        ),
    
        # remove blue page background
        paper_bgcolor="white",
        plot_bgcolor="white",
    
        width=900, height=900,
        margin=dict(l=0, r=0, t=0, b=0),
    
        # hide the 2D axes created by legend dummies
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )

    

    # --- save (requires 'kaleido'); comment out if not installed
    # fig.write_image(f'Figures/node_surface_{param}.png', scale=3)
    fig.show()


    
def get_star(p):
    if p < 0.001:
        return '✶✶✶'
    elif p < 0.01:
        return '✶✶'
    elif p < 0.05:
        return '✶'
    else:
        return ''



def ParticipationCoeff(thr_FC, thr_sc, conn_mat, SCconn_allsub, sub_size):

    COST = 1  
    methods = {
        'infomap': lambda g: g.community_infomap(edge_weights='weight'),
        'louvain': lambda g: g.community_multilevel(weights='weight'),
        'walktrap': lambda g: g.community_walktrap(weights='weight').as_clustering(),
        'label_prop': lambda g: g.community_label_propagation(weights='weight'),
    }
    
    methods_labels = ['infomap', 'louvain', 'walktrap', 'label_prop']
    
    #### adjacency matrix definition/cleaning ###############
    
    
    SCmat_th = np.zeros((SCconn_allsub.shape[0], SCconn_allsub.shape[1], sub_size))
    FCconn_th = np.zeros((conn_mat.shape[0], conn_mat.shape[1], sub_size))
    
    for i in range(sub_size):
        SCmat_th[:, :, i] = threshold_proportional(SCconn_allsub[:, :, i], thr_sc)
        FCconn_th[:, :, i] = threshold_proportional(conn_mat[:, :, i], thr_FC)
    
    SC_mask = SCmat_th.copy()
    SC_mask[SC_mask > 0] = 1
    
    
    FCmat_SC = np.multiply(FCconn_th , SC_mask)
    FCmat_SC = FCconn_th
       
    adj_matrix = FCmat_SC
    
          
    n_subjects = adj_matrix.shape[2]
    n_nodes = adj_matrix.shape[0]
    
    
    all_pc = {m: np.zeros((n_nodes, n_subjects)) for m in methods}
    all_wmd = {m: np.zeros((n_nodes, n_subjects)) for m in methods}
    modularities = {m: np.zeros(n_subjects) for m in methods}
    
    
    for i, sun in enumerate(range(sub_size)):
        
        fc_matrix = adj_matrix[:, :, i]
        fc_matrix = np.nan_to_num(fc_matrix, nan= 0)
        fc_matrix = (fc_matrix + fc_matrix.T) / 2
        np.fill_diagonal(fc_matrix, 0)
    
        
        graph = matrix_to_igraph(fc_matrix, cost=COST)
    
        for method, detect_fn in methods.items():
           # print(method)
            #print(detect_fn)
            community_object = detect_fn(graph)
            modularities[method][i] = community_object.modularity
    
            bg = brain_graph(community_object)
            all_pc[method][:, i] = bg.pc
            all_wmd[method][:, i] = bg.wmd
    

    return all_pc


def matrix_to_igraph(matrix, cost=0.1):
    """
    Converts a functional connectivity matrix to an igraph object.
    Applies a proportional threshold based on the 'cost' parameter.
    """
    # Ensure the matrix is symmetric
    matrix = np.array(matrix)
    if not np.allclose(matrix, matrix.T):
        raise ValueError("Input matrix must be symmetric.")

    # Zero out the diagonal
    np.fill_diagonal(matrix, 0)

    # Flatten the upper triangle to get all possible edges
    triu_indices = np.triu_indices_from(matrix, k=1)
    weights = matrix[triu_indices]

    # Determine the threshold to retain the top 'cost' proportion of weights
    n_edges = int(np.floor(cost * len(weights)))
    if n_edges < 1:
        raise ValueError("Cost too low; no edges retained.")
    threshold = np.sort(weights)[-n_edges]

    # Apply threshold
    thresholded_matrix = np.where(matrix >= threshold, matrix, 0)

    # Create igraph object
    g = ig.Graph.Weighted_Adjacency(thresholded_matrix.tolist(), mode=ig.ADJ_UNDIRECTED, attr="weight")
    return g


class brain_graph:
    def __init__(self, community_object):
        """
        Initializes the brain_graph object with community detection results.
        Computes Participation Coefficient (PC) and Within-Module Degree (WMD).
        """
        self.graph = community_object.graph
        self.community = community_object
        self.pc = self.compute_pc()
        self.wmd = self.compute_wmd()

    def compute_pc(self):
        """
        Computes the Participation Coefficient for each node.
        """
        membership = self.community.membership
        pc = []
        for i in range(self.graph.vcount()):
            neighbors = self.graph.neighbors(i)
            ki = sum(self.graph.es[self.graph.get_eid(i, j)]["weight"] for j in neighbors)
            if ki == 0:
                pc.append(0.0)
                continue
            comm_weights = {}
            for j in neighbors:
                comm = membership[j]
                weight = self.graph.es[self.graph.get_eid(i, j)]["weight"]
                comm_weights[comm] = comm_weights.get(comm, 0) + weight
            pc_val = 1 - sum((w / ki) ** 2 for w in comm_weights.values())
            pc.append(pc_val)
        return pc

    def compute_wmd(self):
        """
        Computes the Within-Module Degree z-score for each node.
        """
        membership = self.community.membership
        degrees = np.array(self.graph.strength(weights="weight"))
        wmd = []
        for i in range(self.graph.vcount()):
            comm = membership[i]
            indices = [idx for idx, m in enumerate(membership) if m == comm]
            comm_degrees = degrees[indices]
            if len(comm_degrees) <= 1:
                wmd.append(0.0)
                continue
            mean = np.mean(comm_degrees)
            std = np.std(comm_degrees)
            if std == 0:
                wmd.append(0.0)
            else:
                wmd.append((degrees[i] - mean) / std)
        return wmd


def only_IC_avg(FCmat, pet, SCmat, thr_sc, thr_FC, sub_size, scmask, nrois_rem):   
    
    SCmat_th = np.zeros((SCmat.shape[0], SCmat.shape[1], sub_size))
    for i in range(sub_size):
        SCmat_th[:, :, i] = threshold_proportional(SCmat[:, :, i], thr_sc)

    SC_mask = SCmat_th.copy()
    SC_mask[SC_mask > 0] = 1


    FCmat_SC = np.multiply(FCmat , SC_mask)


    pet_rem = pet.copy()

    ICallsub_w = np.zeros((nrois_rem, sub_size))

    for j in range(sub_size):
        FCconn = FCmat_SC[:, :, j]
        CMR2 = pet_rem[:, j]
        FCconn_th = threshold_proportional(FCconn, thr_FC)
        E_MAT = np.tile(CMR2, (nrois_rem, 1)) #########

        

        #row_sums = np.sum(FCconn_th, axis=1)  # Compute the sum of each row
       # FCconn_thnor = FCconn_th / row_sums[:, np.newaxis]  # Divide each element by the corresponding row sum
     
        
        row_sums = np.nansum(FCconn_th, axis=1)  # Compute the sum of each row
        FCconn_thnor = np.divide(FCconn_th, row_sums[:, np.newaxis], out=np.zeros_like(FCconn_th), where=row_sums[:, np.newaxis] != 0)

        # FCconn_thnor = np.zeros((nrois_rem, nrois_rem))
       

       # for r in range(nrois_rem):
          #  for s in range(nrois_rem):
           #     FCconn_thnor[r, s] = np.divide(FCconn_th[r, s] , np.nansum(FCconn_th[r, :]))
                
        
        IC_w = np.nansum(np.multiply(FCconn_thnor , E_MAT), axis=1)

        ICallsub_w[:, j] = IC_w

    ICallsub_w[ICallsub_w == 0] = np.nan
    ICallsub_w_avg = np.nanmean(ICallsub_w, axis=1)
    
    return ICallsub_w_avg



# In[6]:




# In[ ]:




