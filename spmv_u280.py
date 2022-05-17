import subprocess
import numpy as np
import argparse
import gen_vectors
import gen_signature
from scipy import sparse
import os
import shutil

def use_fpga(data):
    # initilize directory name for sparse and dense features
    data_path = ['vec_dat', 'sig_dat']
    host_path = './build_dir.hw.xilinx_u280_xdma_201920_3'
    xclbin_path = './build_dir.hw.xilinx_u280_xdma_201920_3'
    sig_path = './sig_dat'
    vec_path = './vec_dat'
    num_runs = 1
    device_id = 0
    dot_product = np.array([[]])
    # set directory path
    parent_dir = os.getcwd()
    # remove existing partitioned data
    remove_data(parent_dir, data_path)
    # process data and partition
    if len(data.shape) == 3: 
        # process 3D tensor data
        for i in range(data.shape[0]):
            process_data(data[i], i, parent_dir)    
    else:
        i = 0
        # process 2D tensor data
        process_data(data, i, parent_dir)
    # sparse library kernels call
    returncode = offload_tofpga(host_path, xclbin_path, sig_path, vec_path, num_runs, device_id)
    # read output vector after SpMV
    # if returncode == 0:
    if len(data.shape) == 3:
        # preprocess for 3D tensor output
        for i in range(data.shape[0]):
            out_vec = read_output(parent_dir+'/vec_dat', i)
            dot_product = np.append(out_vec[:,:data.shape[2]])
    else:
        # preprocess 2D tensor output
        i = 0
        out_vec = read_output(parent_dir+'/vec_dat', i) 
        dot_product = np.append(out_vec)
    # else:
        # print("SpMV Failed in U280")        
    return dot_product

def remove_data(parent_dir, data_path):
    for i in range(len(data_path)):
        for filename in os.listdir(parent_dir+'/'+data_path[i]):
            file_path = os.path.join(parent_dir+'/'+data_path[i], filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

def process_data(data, i, parent_dir):
    features = data
    features.numpy()
    # build symmetric zero matrices 
    if features.shape[0] > features.shape[1]:
        padded_data = np.zeros((features.shape[0],features.shape[0]))
    else:
        padded_data = np.zeros((features.shape[1],features.shape[1]))    
    # zero padding the data for symmetric matrices
    padded_data[:features.shape[0],:features.shape[1]] = features[:,:]    
    # taking sparse and dense features
    denseFeatures = np.transpose(padded_data)
    spFeatures = padded_data
    # process dense vector
    process_denseVector(denseFeatures, i, parent_dir+'/vec_dat')
    # process sparse matrix
    process_sparseMatrix(spFeatures, i, parent_dir+'/sig_dat')

def process_denseVector(denseFeatures, i, parent_dir):
    if denseFeatures.shape[0] == denseFeatures.shape[1]:
        # pass dense vector to get COO format data
        denseVector = sparse.coo_matrix(denseFeatures)
        # partition dense vector
        gen_vectors.process_matrices(denseVector, i, parent_dir)

def process_sparseMatrix(spFeatures, i, parent_dir):
    if spFeatures.shape[0] == spFeatures.shape[1]:
        # Pass dense vector to get COO format data
        sparseMtx = sparse.coo_matrix(spFeatures)
        # partition sparse matrix
        gen_signature.process_matrices(sparseMtx, i, parent_dir)

def offload_tofpga(hostExePath, xclbinPath, sigPath, vecPath, numRuns, deviceId):
    hostExe = os.path.join(hostExePath, "host.exe")
    xclbin = os.path.join(xclbinPath, "spmv.xclbin")
    sigPath.strip()
    sigPath.rstrip("/")
    vecPath.strip()
    vecPath.rstrip("/")
    mtxNames = os.listdir(sigPath)
    mtxNames.sort()
    numErrs = 0
    for mtxName in mtxNames:
        exitCode, outTxt, errTxt  = runApp(hostExe, xclbin, sigPath, vecPath, mtxName, str(numRuns), str(deviceId))
        if (exitCode == 0):
            for item in outTxt.split("\n"):
                if "DATA_CSV:," in item:
                    item = item.lstrip("DATA_CSV:,")
                    print(item+"\n")
        else:
            print("Stdout: "+outTxt)
            print("Stderr: "+errTxt)
            numErrs += 1
    if (numErrs == 0):
        print ("All computations done!")
    else:
        print("Some tests failed! Total {} failed tests!".format(numErrs))
    return exitCode

def runApp(hostExe, xclbin, sigPath, vecPath, mtxName, numRuns, deviceId):
    res = subprocess.run([hostExe, xclbin, sigPath, vecPath, mtxName, numRuns, deviceId], universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return res.returncode, res.stdout, res.stderr

def read_output(outvec_path, i):
    mtx_nam = '/Mtx' + str(i).zfill(3)
    file_name = '/outVec.dat'
    try: 
        out_vec = np.fromfile(outvec_path+mtx_nam+file_name, dtype= float)
        return out_vec
    except IOError as e:
        print(e)
    

def main(args):
    if (args.usage):
        print('Usage example:')
        print('--fileName "dac_sample.txt"')
    else:
        print('--Use API call in application')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs Recommeder Systems sample model using kaggle data')
    parser.add_argument('--usage',action='store_true',help='print usage example')
    parser.add_argument('--fileName',action='store_true',help='print usage example')
    parser.add_argument('--host_path',type=str,default='./build_dir.hw.xilinx_u280_xdma_201920_3',help='directory for host executable')
    parser.add_argument('--xclbin_path',type=str,default='./build_dir.hw.xilinx_u280_xdma_201920_3',help='directory for xclbin file')
    parser.add_argument('--sig_path',type=str,default='./sig_dat',help='directory for matrix signature data')
    parser.add_argument('--vec_path',type=str,default='./vec_dat',help='directory for vector data')
    parser.add_argument('--num_runs',type=int,default=1,help='number of times that SPMV operations is carried out in hardware')
    parser.add_argument('--device_id',type=int,default=0,help='device ID for the U280 FPGA card')
    args = parser.parse_args()
    main(args)