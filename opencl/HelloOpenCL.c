// headers
#include <stdio.h>
#include <stdlib.h> // exit()
#include <string.h> // strlen()

#include <OpenCL/OpenCL.h> // in windows & linux : #include <CL/opencl.h>

// global OpenCL variables
cl_int ret_ocl;
cl_platform_id oclPlatformID;
cl_device_id oclComputeDeviceID; // compute device id
cl_context oclContext; // compute context
cl_command_queue oclCommandQueue; // compute command queue
cl_program oclProgram; // compute program
cl_kernel oclKernel; // compute kernel

char *oclSourceCode=NULL;
size_t sizeKernelCodeLength;

float *hostInput1=NULL;
float *hostInput2=NULL;
float *hostOutput=NULL;

cl_mem deviceInput1=NULL;
cl_mem deviceInput2=NULL;
cl_mem deviceOutput=NULL;

int main(void)
{
    // function declarations
    void cleanup(void);
    char* loadOclProgramSource(const char *,const char *,size_t *);
    
    // variable declarations
    int inputLength;
   
    // code
    
    // hard coded host vectors length
    inputLength=5;
   
    // allocate host-memory
    hostInput1=(float *)malloc(inputLength * sizeof(float));
    if(hostInput1== NULL)
    {
        printf("CPU Memory Fatal Error = Can Not Allocate Enough Memory For Host Input Array 1.\nExitting ...\n");
        exit(EXIT_FAILURE);
    }

    hostInput2=(float *)malloc(inputLength * sizeof(float));
    if(hostInput2== NULL)
    {
        printf("CPU Memory Fatal Error = Can Not Allocate Enough Memory For Host Input Array 2.\nExitting ...\n");
        free(hostInput1);
        exit(EXIT_FAILURE);
    }

    // allocate host-memory to hold 'float' type host vector hostOutput
    hostOutput=(float *)malloc(inputLength * sizeof(float));
    if(hostOutput== NULL)
    {
        printf("CPU Memory Fatal Error = Can Not Allocate Enough Memory For Host Output Array.\nExitting ...\n");
        free(hostInput1);
        free(hostInput2);
        exit(EXIT_FAILURE);
    }

    // fill above input host vectors with arbitary but hard-coded data
    hostInput1[0]=101.0;
    hostInput1[1]=102.0;
    hostInput1[2]=103.0;
    hostInput1[3]=104.0;
    hostInput1[4]=105.0;
    
    hostInput2[0]=201.0;
    hostInput2[1]=202.0;
    hostInput2[2]=203.0;
    hostInput2[3]=204.0;
    hostInput2[4]=205.0;
    
    // get OpenCL supporting platform's ID
    ret_ocl=clGetPlatformIDs(1,&oclPlatformID,NULL);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clGetDeviceIDs() Failed : %d. Exitting Now ...\n",ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    // get OpenCL supporting GPU device's ID
    ret_ocl=clGetDeviceIDs(oclPlatformID,CL_DEVICE_TYPE_GPU,1,&oclComputeDeviceID,NULL);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clGetDeviceIDs() Failed : %d. Exitting Now ...\n",ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    // create OpenCL compute context
    oclContext=clCreateContext(NULL,1,&oclComputeDeviceID,NULL,NULL,&ret_ocl);
    if(ret_ocl!=CL_SUCCESS)
    {
        printf("OpenCL Error - clCreateContext() Failed : %d. Exitting Now ...\n",ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    // create command queue
    oclCommandQueue=clCreateCommandQueue(oclContext,oclComputeDeviceID,0,&ret_ocl);
    if(ret_ocl!=CL_SUCCESS)
    {
        printf("OpenCL Error - clCreateCommandQueue() Failed : %d. Exitting Now ...\n",ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    // create OpenCL program from .ocl
    const char *szOpenCLKernelPath="VecAdd.cl";
    oclProgram=clCreateProgramWithSource(oclContext, 1, &szOpenCLKernelPath, NULL, &ret_ocl);
    if(ret_ocl!=CL_SUCCESS)
    {
        printf("OpenCL Error - clCreateProgramWithSource() Failed : %d. Exitting Now ...\n",ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    // build OpenCL program
    ret_ocl=clBuildProgram(oclProgram,0,NULL,NULL,NULL,NULL);
    if(ret_ocl!=CL_SUCCESS)
    {
        printf("OpenCL Error - clBuildProgram() Failed : %d. Exitting Now ...\n",ret_ocl);
        
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(oclProgram,oclComputeDeviceID,CL_PROGRAM_BUILD_LOG,sizeof(buffer),buffer,&len);
        printf("OpenCL Program Build Log : %s\n",buffer);
        
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    // create OpenCL kernel by passing kernel function name that we used in .cl file
    oclKernel=clCreateKernel(oclProgram,"vecAdd",&ret_ocl);
    if(ret_ocl!=CL_SUCCESS)
    {
        printf("OpenCL Error - clCreateKernel() Failed : %d. Exitting Now ...\n",ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    int size=inputLength * sizeof(cl_float);
    // allocate device-memory
    deviceInput1=clCreateBuffer(oclContext,CL_MEM_READ_ONLY,size,NULL,&ret_ocl);
    if(ret_ocl!=CL_SUCCESS)
    {
        printf("OpenCL Error - clCreateBuffer() Failed For 1st Input Array : %d. Exitting Now ...\n",ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    deviceInput2=clCreateBuffer(oclContext,CL_MEM_READ_ONLY,size,NULL,&ret_ocl);
    if(ret_ocl!=CL_SUCCESS)
    {
        printf("OpenCL Error - clCreateBuffer() Failed For 2nd Input Array : %d. Exitting Now ...\n",ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    deviceOutput=clCreateBuffer(oclContext,CL_MEM_WRITE_ONLY,size,NULL,&ret_ocl);
    if(ret_ocl!=CL_SUCCESS)
    {
        printf("OpenCL Error - clCreateBuffer() Failed For 2nd Input Array : %d. Exitting Now ...\n",ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    // set OpenCL kernel arguments. Our OpenCL kernel has 4 arguments 0,1,2,3
    // set 0 based 0th argument i.e. deviceInput1
    ret_ocl=clSetKernelArg(oclKernel,0,sizeof(cl_mem),(void *)&deviceInput1); // 'deviceInput1' maps to 'in1' param of kernel function in .cl file
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clSetKernelArg() Failed For 1st Argument : %d. Exitting Now ...\n",ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    // set 0 based 1st argument i.e. deviceInput2
    ret_ocl=clSetKernelArg(oclKernel,1,sizeof(cl_mem),(void *)&deviceInput2); // 'deviceInput2' maps to 'in2' param of kernel function in .cl file
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clSetKernelArg() Failed For 2nd Argument : %d. Exitting Now ...\n",ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    // set 0 based 2nd argument i.e. deviceOutput
    ret_ocl=clSetKernelArg(oclKernel,2,sizeof(cl_mem),(void *)&deviceOutput); // 'deviceOutput' maps to 'out' param of kernel function in .cl file
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clSetKernelArg() Failed For 3rd Argument : %d. Exitting Now ...\n",ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    // set 0 based 3rd argument i.e. len
    ret_ocl=clSetKernelArg(oclKernel,3,sizeof(cl_int),(void *)&inputLength); // 'inputLength' maps to 'len' param of kernel function in .cl file
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clSetKernelArg() Failed For 4th Argument : %d. Exitting Now ...\n",ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    // write abve 'input' device buffer to device memory
    ret_ocl=clEnqueueWriteBuffer(oclCommandQueue,deviceInput1,CL_FALSE,0,size,hostInput1,0,NULL,NULL);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clEnqueueWriteBuffer() Failed For 1st Input Device Buffer : %d. Exitting Now ...\n",ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    ret_ocl=clEnqueueWriteBuffer(oclCommandQueue,deviceInput2,CL_FALSE,0,size,hostInput2,0,NULL,NULL);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clEnqueueWriteBuffer() Failed For 2nd Input Device Buffer : %d. Exitting Now ...\n",ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    // run the kernel
    size_t global_size=5; // 1-D array size
    // 1 in following argument list is for N-D, i.e. N=1 means 1-Dimension
    ret_ocl=clEnqueueNDRangeKernel(oclCommandQueue,oclKernel,1,NULL,&global_size,NULL,0,NULL,NULL);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clEnqueueNDRangeKernel() Failed : %d. Exitting Now ...\n",ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    // finish OpenCL command queue
    clFinish(oclCommandQueue);
    
    // read back result from the device (i.e from deviceOutput) into cpu variable (i.e hostOutput)
    ret_ocl=clEnqueueReadBuffer(oclCommandQueue,deviceOutput,CL_TRUE,0,size,hostOutput,0,NULL,NULL);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clEnqueueReadBuffer() Failed : %d. Exitting Now ...\n",ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // results
    int i;
    for(i=0;i<inputLength;i++)
    {
        printf("%f + %f = %f\n",hostInput1[i],hostInput2[i],hostOutput[i]);
    }

    // total cleanup
    cleanup();

    return(0);
}

void cleanup(void)
{
    // code
    
    // OpenCL cleanup
    if(oclSourceCode)
    {
        free((void *)oclSourceCode);
        oclSourceCode=NULL;
    }
    
    if(oclKernel)
    {
        clReleaseKernel(oclKernel);
        oclKernel=NULL;
    }
    
    if(oclProgram)
    {
        clReleaseProgram(oclProgram);
        oclProgram=NULL;
    }
    
    if(oclCommandQueue)
    {
        clReleaseCommandQueue(oclCommandQueue);
        oclCommandQueue=NULL;
    }
    
    if(oclContext)
    {
        clReleaseContext(oclContext);
        oclContext=NULL;
    }

    // free allocated device-memory
    if(deviceInput1)
    {
        clReleaseMemObject(deviceInput1);
        deviceInput1=NULL;
    }
    
    if(deviceInput2)
    {
        clReleaseMemObject(deviceInput2);
        deviceInput2=NULL;
    }
    
    if(deviceOutput)
    {
        clReleaseMemObject(deviceOutput);
        deviceOutput=NULL;
    }
    
    // free allocated host-memory
    if(hostInput1)
    {
        free(hostInput1);
        hostInput1=NULL;
    }
    
    if(hostInput2)
    {
        free(hostInput2);
        hostInput2=NULL;
    }
    
    if(hostOutput)
    {
        free(hostOutput);
        hostOutput=NULL;
    }
}

char* loadOclProgramSource(const char *filename, const char *preamble, size_t *sizeFinalLength)
{
    // locals
    FILE *pFile=NULL;
    size_t sizeSourceLength;
    
    pFile=fopen(filename,"rb"); // binary read
    if(pFile==NULL)
        return(NULL);
    
    size_t sizePreambleLength=(size_t)strlen(preamble);
    
    // get the length of the source code
    fseek(pFile,0,SEEK_END);
    sizeSourceLength=ftell(pFile);
    fseek(pFile,0,SEEK_SET); // reset to beginning
    
    // allocate a buffer for the source code string and read it in
    char *sourceString=(char *)malloc(sizeSourceLength+sizePreambleLength+1);
    memcpy(sourceString, preamble, sizePreambleLength);
    if(fread((sourceString)+sizePreambleLength,sizeSourceLength,1,pFile)!=1)
    {
        fclose(pFile);
        free(sourceString);
        return(0);
    }
    
    // close the file and return the total length of the combined (preamble + source) string
    fclose(pFile);
    if(sizeFinalLength != 0)
    {
        *sizeFinalLength = sizeSourceLength + sizePreambleLength;
    }
    sourceString[sizeSourceLength + sizePreambleLength]='\0';
    
    return(sourceString);
}
