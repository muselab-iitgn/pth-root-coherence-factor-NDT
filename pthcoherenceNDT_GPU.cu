#include "mex.h"
#include "cuda_runtime.h"
#include <cmath>

__global__ void pthCoherenceKernel(double* d_BeamformData, const double* d_RData,
    const double* d_element_Pos, double speed_Of_Sound_umps,
    double RF_Start_Time, double fs,
    const double* d_BeamformX, const double* d_BeamformZ,
    const double* d_element_loc, double p,
    int num_samples, int num_rx_elements, int nX, int nZ)
{
    int Xi = blockIdx.x * blockDim.x + threadIdx.x;
    int Zi = blockIdx.y * blockDim.y + threadIdx.y;

    if (Xi >= nX || Zi >= nZ) return;

    double px = d_BeamformX[Xi];
    double pz = d_BeamformZ[Zi];
    double tx_x = d_element_loc[0];

    double das_sum = 0.0;
    double p_root_sum = 0.0;
    double sum_sq = 0.0;

    for (int ex = 0; ex < num_rx_elements; ex++) {
        double rx_x = d_element_Pos[ex];
        double dist_tx_pixel = sqrt((px - tx_x) * (px - tx_x) + pz * pz);
        double dist_pixel_rx = sqrt((px - rx_x) * (px - rx_x) + pz * pz);
        double distance_Along_RF = dist_tx_pixel + dist_pixel_rx;
        double time_Pt_Along_RF = distance_Along_RF / speed_Of_Sound_umps;
        int samples = static_cast<int>(floor((time_Pt_Along_RF - RF_Start_Time) * fs + 0.5));

        if (samples >= 0 && samples < num_samples) {
            double s_i = d_RData[samples + ex * num_samples];
            das_sum += s_i;
            double val_p_root = pow(fabs(s_i), 1.0 / p);
            p_root_sum += (s_i < 0.0) ? -val_p_root : val_p_root;
            sum_sq += s_i * s_i;
        }
    }

    double nR_val = pow(fabs(p_root_sum), p);
    double nR = (p_root_sum < 0.0) ? -nR_val : nR_val;
    double Nr = nR * nR;
    double Dr = sum_sq;
    double pCF = 0.0;

    if (Dr > 1e-12) {
        pCF = (1.0 / num_rx_elements) * (Nr / Dr);
    }

    d_BeamformData[Xi + Zi * nX] = das_sum * pCF;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 9) {
        mexErrMsgIdAndTxt("CUDA:pthcoherenceNDT:nrhs", "Nine inputs required.");
    }
    if (nlhs != 1) {
        mexErrMsgIdAndTxt("CUDA:pthcoherenceNDT:nlhs", "One output required.");
    }

    for (int i = 0; i < nrhs; i++) {
        if (!mxIsDouble(prhs[i]) || mxIsComplex(prhs[i])) {
            mexErrMsgIdAndTxt("CUDA:pthcoherenceNDT:notDouble", "All inputs must be non-complex double arrays.");
        }
    }

    const double *h_RData = (const double*)mxGetData(prhs[0]);
    int num_samples = (int)mxGetM(prhs[0]);
    int num_rx_elements = (int)mxGetN(prhs[0]);

    const double *h_element_Pos = (const double*)mxGetData(prhs[1]);
    int num_pos = (int)mxGetNumberOfElements(prhs[1]);
    if (num_pos != num_rx_elements) {
        mexErrMsgIdAndTxt("CUDA:pthcoherenceNDT:dimMismatch", "Number of elements in RData and element_Pos must match.");
    }

    double speed_Of_Sound_umps = mxGetScalar(prhs[2]);
    double RF_Start_Time = mxGetScalar(prhs[3]);
    double fs = mxGetScalar(prhs[4]);

    const double *h_BeamformX = (const double*)mxGetData(prhs[5]);
    int nX = (int)mxGetNumberOfElements(prhs[5]);

    const double *h_BeamformZ = (const double*)mxGetData(prhs[6]);
    int nZ = (int)mxGetNumberOfElements(prhs[6]);

    const double *h_element_loc = (const double*)mxGetData(prhs[7]);
    if (mxGetNumberOfElements(prhs[7]) < 1) {
        mexErrMsgIdAndTxt("CUDA:pthcoherenceNDT:badLoc", "element_loc must have at least one value.");
    }

    double p = mxGetScalar(prhs[8]);

    plhs[0] = mxCreateNumericMatrix(nX, nZ, mxDOUBLE_CLASS, mxREAL);
    double *h_BeamformData = (double*)mxGetData(plhs[0]);

    double *d_RData, *d_element_Pos, *d_BeamformX, *d_BeamformZ, *d_element_loc, *d_BeamformData;
    cudaError_t err;

    err = cudaMalloc(&d_RData, num_samples * num_rx_elements * sizeof(double));
    if(err != cudaSuccess) mexErrMsgTxt(cudaGetErrorString(err));

    err = cudaMalloc(&d_element_Pos, num_rx_elements * sizeof(double));
    if(err != cudaSuccess) mexErrMsgTxt(cudaGetErrorString(err));

    err = cudaMalloc(&d_BeamformX, nX * sizeof(double));
    if(err != cudaSuccess) mexErrMsgTxt(cudaGetErrorString(err));

    err = cudaMalloc(&d_BeamformZ, nZ * sizeof(double));
    if(err != cudaSuccess) mexErrMsgTxt(cudaGetErrorString(err));

    err = cudaMalloc(&d_element_loc, 2 * sizeof(double));
    if(err != cudaSuccess) mexErrMsgTxt(cudaGetErrorString(err));

    err = cudaMalloc(&d_BeamformData, nX * nZ * sizeof(double));
    if(err != cudaSuccess) mexErrMsgTxt(cudaGetErrorString(err));

    cudaMemcpy(d_RData, h_RData, num_samples * num_rx_elements * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_element_Pos, h_element_Pos, num_rx_elements * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_BeamformX, h_BeamformX, nX * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_BeamformZ, h_BeamformZ, nZ * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_element_loc, h_element_loc, 2 * sizeof(double), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((nX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (nZ + threadsPerBlock.y - 1) / threadsPerBlock.y);

    pthCoherenceKernel<<<numBlocks, threadsPerBlock>>>(
        d_BeamformData, d_RData, d_element_Pos,
        speed_Of_Sound_umps, RF_Start_Time, fs,
        d_BeamformX, d_BeamformZ, d_element_loc, p,
        num_samples, num_rx_elements, nX, nZ
    );

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        mexErrMsgIdAndTxt("CUDA:pthcoherenceNDT:kernelError", cudaGetErrorString(err));
    }

    cudaMemcpy(h_BeamformData, d_BeamformData, nX * nZ * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_RData);
    cudaFree(d_element_Pos);
    cudaFree(d_BeamformX);
    cudaFree(d_BeamformZ);
    cudaFree(d_element_loc);
    cudaFree(d_BeamformData);
}
