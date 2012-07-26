# Mirror is done inside the kernel

import numpy
import time
import scipy.ndimage as nd

def gpuSwitchtime(StackImages,dim_x,dim_y,dim_z,usekernel,device=None):
	"""
	Return a matrix with the positions of a step in a sequence for each pixel

	Parameters:
	---------------
	StackImages: int32 : 3D Array of images

	useKernel : string
		step = [1]*5 +[-1]*5
		zero = [1]*5 +[0] + [-1]*5

	dim_x= x-dimensions of images
	dim_y= y-dimensions of images
	dim_z= z-dimensions of images

	device: Set the GPU device to use (numbered from 0). Default is 0
	"""

	t1=time.time()
	if device is None:
		CUDA_DEVICE=0
	else:
		CUDA_DEVICE=device


	import pycuda.driver as cuda
	import pycuda.autoinit
	from pycuda.gpuarray import to_gpu
	from pycuda.compiler import SourceModule

	if usekernel =="step":
		kernel=[1]*5+[-1]*5
		kernel2=kernel[::-1]
		kernel2=numpy.array(kernel2,dtype=numpy.int32)
		if (len(kernel)%2==0):
			origin=-1 
		else:
			origin=0
	if usekernel =="zero":
		kernel=[1]*5 +[0] + [-1]*5
		kernel2=kernel[::-1]
		kernel2=numpy.array(kernel2,dtype=numpy.int32)
		if (len(kernel)%2==0):
			origin=-1
		else:
			origin=0


	stepsize=len(kernel)
	t1=time.time()
	aMod = numpy.zeros(((dim_z+stepsize-1),dim_y,dim_x),dtype=numpy.int32)
	switch=numpy.zeros((dim_y,dim_x),dtype=numpy.int32)
	mod = SourceModule("""
	__global__ void findconvolve1d(int *stack_gpu,int *kernel_gpu ,int *amod,int dim_x, int dim_y, int dim_z,int step_size,int origine)
	{
	  int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	  int idy = threadIdx.y + blockIdx.y * blockDim.y; 
	  if (idx >= dim_x || idy >= dim_y)
	    return;
	  int j,idz,id,id1;

		//Copio elementi nel secondo stack per il mirror dei primi e ultimi elementi
		for(idz=0;idz<dim_z;idz++)
		{
			int flat_id = idx + dim_x * idy + (dim_x * dim_y) * idz	;
			int flat_id2= idx + dim_x * idy + (dim_x * dim_y) * (idz+step_size/2+origine);
			amod[flat_id2]=stack_gpu[flat_id];
		}

		//Mirror dei primi elementi
		int idz2=step_size/2-1+origine;
		for(id=0;id<step_size/2+origine;id++)
		{
		
			int flat_id = idx + dim_x * idy + (dim_x * dim_y) * idz2;		
			int flat_id3= idx + dim_x * idy + (dim_x * dim_y) * id;	
			amod[flat_id3]=stack_gpu[flat_id];
			idz2--;
		}
		//Mirror ultimi elementi
		int idz3=dim_z-1;
		for(id1=dim_z+step_size/2+origine;id1<dim_z+step_size-1;id1++)
		{
			int flat_id4= idx + dim_x * idy + (dim_x * dim_y) * id1;
			int flat_id5= idx + dim_x * idy + (dim_x * dim_y) * idz3;	
			amod[flat_id4]=stack_gpu[flat_id5];
			idz3--;

		}



		for(idz = 0; idz <dim_z; idz++)
		{
			int flat_id8 = idx + dim_x * idy + (dim_x * dim_y) * idz; 
			stack_gpu[flat_id8]=0;
		}	



		for(idz=step_size/2+origine;idz<dim_z+step_size/2+origine;idz++)
		{
		
			int flat_id6 = idx + dim_x * idy + (dim_x * dim_y) * (idz-step_size/2-origine);  	
	
			for(j=0;j<step_size;j++)
			{
				int flat_id7 = idx + dim_x * idy + (dim_x * dim_y) * (idz-step_size/2-origine+j); 
				stack_gpu[flat_id6]+=amod[flat_id7]*kernel_gpu[j];
			}
		} 



	}
	__global__ void findmin(int *stack_gpu,int *switch_gpu,int dim_x, int dim_y, int dim_z)
	{
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	int idy = threadIdx.y + blockIdx.y * blockDim.y; 
	  if (idx >= dim_x || idy >= dim_y)
	    return;
	int flat_id1 = idx + dim_x * idy ;
	int min=4294967295;
	for(int idz = 0; idz <dim_z; idz++)
	  {
		int flat_id = idx + dim_x * idy + (dim_x * dim_y) * idz;      
		if(stack_gpu[flat_id]<min)
		{
		min=stack_gpu[flat_id];
		switch_gpu[flat_id1]=idz;
		}
	  }
	}
	""")
	block_size = 32 
	func = mod.get_function("findconvolve1d")
	func2=mod.get_function("findmin") #Crea la matrice con il tempo in cui aviene lo switch





	#Host to Device copy
	stack_gpu=to_gpu(StackImages)
	kernel_gpu=to_gpu(kernel2)
	switch_gpu=to_gpu(switch)
	aMod_gpu=to_gpu(aMod)

	#Function calls
	func(stack_gpu,kernel_gpu, aMod_gpu,numpy.int32(dim_x), numpy.int32(dim_y), numpy.int32(dim_z),numpy.int32(stepsize),numpy.int32(origin),
	  block=(block_size,block_size,1),
	  grid=((dim_x - 1) / block_size + 1,(dim_y - 1) / block_size + 1))

	func2(stack_gpu,switch_gpu,numpy.int32(dim_x), numpy.int32(dim_y), numpy.int32(dim_z),block=(block_size,block_size,1),
	grid=((dim_x - 1) / block_size + 1,(dim_y - 1) / block_size + 1))

	#Device to host copy
	switch=switch_gpu.get()


	#Free GPU memory
	stack_gpu.gpudata.free()
	switch_gpu.gpudata.free()
	aMod_gpu.gpudata.free()

	tgpu=time.time()-t1
	print ("GPU Calculus done in %.4f s" %tgpu)

	return switch,tgpu


if __name__ == "__main__":


	dim_x=200
	dim_y=200
	dim_z=200
	a = numpy.random.randn(dim_z,dim_y,dim_x)
	a = a.astype(numpy.int32)
	results=gpuSwitchtime(a,dim_x,dim_y,dim_z,usekernel="step",device=0)	
	gpuswitch=results[0]
	step=step = [1]*5 +[-1]*5
	#zero=[1]*5 +[0]+[-1]*5
	cpuswitch=numpy.zeros((dim_y,dim_x),dtype=numpy.int32)
	t3=time.time()
	for i in range(0,dim_x):
	    for j in range(0,dim_y):
	       indice=(nd.convolve1d(a[:,j,i],step,mode='reflect')).argmin()      
	       cpuswitch[j,i]=indice


	t4=time.time()
	tcpu=t4-t3
	print ("CPU calculus done = %.4f s" %tcpu)
	print "Difference : \n"
	print gpuswitch-cpuswitch
	print ("\nGPU is %d times faster than CPU " %(tcpu/results[1]))
