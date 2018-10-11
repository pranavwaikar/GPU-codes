// OpenCL kernel
__kernel void matrixMultiply(__global int *A, __global int *B, __global int *C,int numARows,int numAColumns,int numBRows,int numBColumns,int numCRows,int numCColumns)
{
    // variable declarations
    int row=get_global_id(0);
    int col=get_global_id(1);
    
    // code
    if((row < numARows) && (col < numBColumns))
    {
        int Cvalue=0;
        for(int k=0; k < numAColumns; k++)
        {
            Cvalue +=A[row * numAColumns + k] * B[k * numBColumns + col];
        }
        C[row * numCColumns + col]=Cvalue;
    }
}
