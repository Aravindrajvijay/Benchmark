import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import os
import shutil
import subprocess
import argparse
import math
from matrix_params import *
from signature import *

def partition_matrix(sparseMtx, mtxName, maxRows, maxCols, channels, parEntries, accLatency, memBits, mtxSigPath):
    l_nnzFileNames = []
    for i in range(channels):
        l_nnzFileNames.append(mtxSigPath+'/'+mtxName+'/nnzVal_' + str(i) + '.dat')

    l_parParamFileName = mtxSigPath+'/'+mtxName+'/parParam.dat'
    l_rbParamFileName  = mtxSigPath+'/'+mtxName+'/rbParam.dat'
    l_infoFileName     = mtxSigPath+'/'+mtxName+'/info.dat'
    l_sig = signature(parEntries, accLatency, channels, maxRows, maxCols, memBits)
    l_sig.process(sparseMtx, mtxName)
    l_sig.store_rbParam(l_rbParamFileName)
    l_sig.store_parParam(l_parParamFileName)
    l_sig.store_nnz(l_nnzFileNames)
    l_sig.store_info(l_infoFileName)
    
def check_signature(mtxName, mtxFullName, maxRows, maxCols, channels, parEntries, accLatency, memBits, mtxSigPath):
    l_nnzFileNames = []
    for i in range(channels):
        l_nnzFileNames.append(mtxSigPath+'/nnzVal_' + str(i) + '.dat')

    l_parParamFileName = mtxSigPath+'/parParam.dat'
    l_rbParamFileName = mtxSigPath+'/rbParam.dat'
    l_infoFileName = mtxSigPath+'/info.dat'
    l_sig = signature(parEntries, accLatency, channels, maxRows, maxCols, memBits)
    l_sig.load_rbParam(l_rbParamFileName)
    l_sig.load_parParam(l_parParamFileName)
    l_sig.load_nnz(l_nnzFileNames)
    l_sig.load_info(l_infoFileName)
    if l_sig.check(mtxFullName, mtxName):
        print("INFO: {} signature verification pass!".format(mtxName))
        return True 
    else:
        print("ERROR: {} signature verification failed!".format(mtxName))
        return False
    
def process_matrices(sparseMtx, i, parent_dir):
    new_dir = 'Mtx' + str(i).zfill(3)
    path = os.path.join(parent_dir, new_dir) 
    os.makedirs(path)
    mtxName    = new_dir
    maxRows    = 4096
    maxCols    = 4096
    channels   = 16
    parEntries = 4
    accLatency = 8
    memBits    = 256
    mtxSigPath = './sig_dat'
    partition_matrix(sparseMtx,mtxName, maxRows, maxCols, channels, parEntries, accLatency, memBits, mtxSigPath)
        

def decode_mtx_sig(mtxSigPath, maxRows, maxCols, channels, parEntries, accLatency, memBits, txtPath):
    if not os.path.exists(mtxSigPath):
        print("ERROR: {} directory doesn't exist.".format(mtxSigPath))
        return
    subprocess.run(["mkdir", "-p", txtPath])
    l_nnzFileNames, l_nnzTxtFileNames = [],[]
    for i in range(channels):
        l_nnzFileNames.append(mtxSigPath+'/nnzVal_' + str(i) + '.dat')
        l_nnzTxtFileNames.append(txtPath+'/nnzVal_' + str(i) + '.txt')

    l_parParamFileName = mtxSigPath+'/parParam.dat'
    l_parParamTxtFileName = txtPath+'/parParam.txt'
    l_rbParamFileName = mtxSigPath+'/rbParam.dat'
    l_rbParamTxtFileName = txtPath+'/rbParam.txt'
    l_infoFileName = mtxSigPath+'/info.dat'
    l_infoTxtFileName = txtPath+'/info.txt'
    l_sig = signature(parEntries, accLatency, channels, maxRows, maxCols, memBits)
    l_sig.load_rbParam(l_rbParamFileName)
    l_sig.print_rbParam(l_rbParamTxtFileName)
    l_sig.load_parParam(l_parParamFileName)
    l_sig.print_parParam(l_parParamTxtFileName)
    l_sig.load_nnz(l_nnzFileNames)
    l_sig.print_nnz(l_nnzTxtFileNames)
    l_sig.load_info(l_infoFileName)
    l_sig.print_info(l_infoTxtFileName)

def main(args):
    if (args.usage):
        print('Usage example:')
        print('python gen_signature.py --partition [--clean] --mtx_list ./test_matrices.txt --sig_path ./sig_dat')
        print('python gen_signature.py --check  --mtx_list ./test_matrices.txt --sig_path ./sig_dat')
        print('python gen_signature.py --decode --mtx_path ./sig_dat/mtx_name  --txt_path ./txt_out/mtx_name')
    elif (args.decode):
        decode_mtx_sig(args.mtx_path, args.max_rows, args.max_cols, args.channels, args.par_entries,args.acc_latency, args.mem_bits, args.txt_path)
    else:
        process_matrices(args.partition, args.clean, args.check, args.mtx_list, args.max_rows, args.max_cols, args.channels, args.par_entries, args.acc_latency, args.mem_bits, args.sig_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='partition sparse matrix, verify partitions and decode partiton info')
    parser.add_argument('--usage',action='store_true',help='print usage example')
    parser.add_argument('--partition',action='store_true',help='partition sparse matrix across HBM channels')
    parser.add_argument('--clean',action='store_true',help='clean up downloaded .mtx file after the run')
    parser.add_argument('--check',action='store_true',help='check partitions against orginial matrices')
    parser.add_argument('--mtx_list',type=str,help='a file containing URLs for downloading sprase matrices')
    parser.add_argument('--max_rows',type=int,default=4096,help='maximum number of rows in each channel block, default value 4096')
    parser.add_argument('--max_cols',type=int,default=4096,help='maximum number of cols in each channel block, default value 4096')
    parser.add_argument('--channels',type=int,default=16,help='number of HBM channels, default value 16')
    parser.add_argument('--par_entries',type=int,default=4,help='number of NNZ entries retrieved from one HBM channel')
    parser.add_argument('--acc_latency',type=int,default=8,help='number of cycles used for double precision accumulation')
    parser.add_argument('--mem_bits',type=int,default=256,help='number of bits in each HBM channel access')
    parser.add_argument('--sig_path',type=str,default='./sig_dat',help='directory for storing partition results, default value ./sig_dat')
    parser.add_argument('--decode',action='store_true',help='print signature files into text files')
    parser.add_argument('--mtx_path',type=str,help='directory for matrix signature data')
    parser.add_argument('--txt_path',type=str,default='./txt_out',help='directory for storing text files, default value ./txt_out')
    args = parser.parse_args()
    main(args)