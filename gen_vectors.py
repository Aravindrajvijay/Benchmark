import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import os
import shutil
import subprocess
import argparse
import math
from matrix_params import *

def write_vec(p_vec, p_vecFileName):
    fo = open(p_vecFileName, "wb")
    p_vec.tofile(fo) 
    fo.close()

def gen_vectors(spm, mtxName, vecPath, parEntries):
    l_inVecFileName = vecPath+'/'+mtxName+'/inVec.dat'
    l_refVecFileName = vecPath+'/'+mtxName+'/refVec.dat'
    l_diagMatFileName = vecPath+'/'+mtxName+'/mat_diag.dat'
    l_nPad = ((spm.n+parEntries-1)//parEntries) * parEntries
    l_mPad = ((spm.m+parEntries-1)//parEntries) * parEntries
    l_inVec = np.zeros(l_nPad, dtype=np.float64)
    l_refVec = np.zeros(l_mPad, dtype=np.float64)
    l_inVec[0:spm.n] = np.random.rand(spm.n).astype(np.float64)
    l_cooMat = sp.coo_matrix((spm.data, (spm.row, spm.col)), shape=(spm.m, spm.n), dtype=np.float64)
    l_refVec[0:spm.m] = l_cooMat.dot(l_inVec[0:spm.n])
    if spm.m == spm.n:
        l_diagVec = np.full(l_mPad, 1, dtype=np.float64)
        for i in range(spm.nnz):
            if spm.row[i] == spm.col[i]:
                l_diagVec[spm.row[i]] = spm.data[i]
        write_vec(l_diagVec, l_diagMatFileName)
    write_vec(l_inVec, l_inVecFileName)
    write_vec(l_refVec, l_refVecFileName)

def process_matrices(denseVec, i, parent_dir):
    parEntries = 4
    new_dir = 'Mtx' + str(i).zfill(3)
    path = os.path.join(parent_dir, new_dir) 
    os.makedirs(path)
    # Set Parameters for the vector data
    vecPath = './vec_dat'
    mtxName = new_dir
    spm = sparse_matrix()
    spm.get_matrix(denseVec,mtxName)
    gen_vectors(spm, mtxName, vecPath, parEntries)
    
def main(args):
    if (args.usage):
        print('Usage example:')
        print('python gen_vectors.py --denseVec --gen_vec [--pcg] [--clean] --vec_path ./vec_dat')
    else:
        process_matrices(args.denseVec)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='read sparse matrix file, generate input vector and golden reference vector for SpMv')
    parser.add_argument('--denseVec',action='store_true',help='print usage example')
    parser.add_argument('--usage',action='store_true',help='print usage example')
    parser.add_argument('--gen_vec',action='store_true',help='generate input and output vectors for a set of sparse matrices')
    parser.add_argument('--pcg',action='store_true',help='generate the vector files required by PCG solver')
    parser.add_argument('--clean',action='store_true',help='clean up downloaded .mtx file after the run')
    parser.add_argument('--par_entries',type=int,default=4,help='number of NNZ entries retrieved from one HBM channel')
    parser.add_argument('--vec_path',type=str,default='./vec_dat',help='directory for storing vectors, default value ./vec_dat')
    args = parser.parse_args()
    main(args)