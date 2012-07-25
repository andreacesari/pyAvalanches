# come la v2 solo che fa l'equivalente della convolve1d (quello che serve a noi)
#######################################################
#               SCRIPT DEFINITIVO                     #
#######################################################

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy
import time
from pycuda.gpuarray import to_gpu
import scipy.ndimage as nd



dim_x=100
dim_y=100
dim_z=500 
kernel= numpy.array([-1,-1,-1,-1,-1,1,1,1,1,1],dtype=numpy.int32)
kernel2=numpy.array(kernel[::-1],dtype=numpy.int32)
origin=-1  # Deve rimanere a -1. (Perche' la convolve1d richiama la correlate1d ma inverte il kernel e pone origine in -1. Io ho implementato la correlate1d sulla quale si basano tutte. Porre a 0 in caso di kernel dispari
step=len(kernel)
t5=time.time()
a = numpy.random.randn(dim_z,dim_y,dim_x)
a = a.astype(numpy.int32)
aMod = numpy.zeros(((dim_z+step-1),dim_y,dim_x),dtype=numpy.int32)
b=numpy.zeros((dim_y,dim_x),dtype=numpy.int32)
b2=numpy.zeros((dim_y,dim_x),dtype=numpy.int32)
t6=time.time()
talloc=t6-t5
mod = SourceModule("""
__global__ void findcorrelation(int *a,int *kernel_gpu ,int *amod,int dim_x, int dim_y, int dim_z,int step1,int origine)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x; //OK
  int idy = threadIdx.y + blockIdx.y * blockDim.y; //OK
  if (idx >= dim_x || idy >= dim_y)
    return;
  int j,idz,id,id1;

	//Copio elementi nel secondo stack per il mirror dei primi e ultimi elementi
	for(idz=0;idz<dim_z;idz++)
	{
		int flat_id = idx + dim_x * idy + (dim_x * dim_y) * idz	;
		int flat_id2= idx + dim_x * idy + (dim_x * dim_y) * (idz+step1/2+origine);
		amod[flat_id2]=a[flat_id];
	}

	//Mirror dei primi elementi
	int idz2=step1/2-1+origine;
	for(id=0;id<step1/2+origine;id++)
	{
		
		int flat_id = idx + dim_x * idy + (dim_x * dim_y) * idz2;	;	
		int flat_id3= idx + dim_x * idy + (dim_x * dim_y) * id;	
		amod[flat_id3]=a[flat_id];
		idz2--;
	}
	//Mirror ultimi elementi
	int idz3=dim_z-1;
	for(id1=dim_z+step1/2+origine;id1<dim_z+step1-1;id1++)
	{
		int flat_id4= idx + dim_x * idy + (dim_x * dim_y) * id1;
		int flat_id5= idx + dim_x * idy + (dim_x * dim_y) * idz3;	
		amod[flat_id4]=a[flat_id5];
		idz3--;

	}



	for(idz = 0; idz <dim_z; idz++)
	{
		int flat_id8 = idx + dim_x * idy + (dim_x * dim_y) * idz; 
		a[flat_id8]=0;
	}	



	//for(idz=step1/2;idz<dim_z+step1/2;idz++)
	for(idz=step1/2+origine;idz<dim_z+step1/2+origine;idz++)
	{
		
		//int flat_id6 = idx + dim_x * idy + (dim_x * dim_y) * (idz-step1/2);  //OK
		int flat_id6 = idx + dim_x * idy + (dim_x * dim_y) * (idz-step1/2-origine);  //OK	
	
		for(j=0;j<step1;j++)
		{
			int flat_id7 = idx + dim_x * idy + (dim_x * dim_y) * (idz-step1/2-origine+j);  //OK
			//int flat_id7 = idx + dim_x * idy + (dim_x * dim_y) * (idz-step1/2+j);  //OK
			a[flat_id6]+=amod[flat_id7]*kernel_gpu[j];
		}
	} 



}
__global__ void findmin(int *a,int *b,int dim_x, int dim_y, int dim_z)
{
int idx = threadIdx.x + blockIdx.x * blockDim.x; //OK
int idy = threadIdx.y + blockIdx.y * blockDim.y; //OK
  if (idx >= dim_x || idy >= dim_y)
    return;
int flat_id1 = idx + dim_x * idy ;
int min=4294967295;
for(int idz = 0; idz <dim_z; idz++)
  {
	int flat_id = idx + dim_x * idy + (dim_x * dim_y) * idz;  //OK     
	if(a[flat_id]<min)
        {
        min=a[flat_id];
        b[flat_id1]=idz;
        }
  }
}
""")
block_size = 32 
func = mod.get_function("findcorrelation")
func2=mod.get_function("findmin") #Crea la matrice con il tempo in cui aviene lo switch
#Begin GPU Call

t1=time.time()
a_gpu=to_gpu(a)
kernel_gpu=to_gpu(kernel2)
b_gpu=to_gpu(b)
aMod_gpu=to_gpu(aMod)

func(a_gpu,kernel_gpu, aMod_gpu,numpy.int32(dim_x), numpy.int32(dim_y), numpy.int32(dim_z),numpy.int32(step),numpy.int32(origin),
  block=(block_size,block_size,1),
  grid=((dim_x - 1) / block_size + 1,(dim_y - 1) / block_size + 1))

func2(a_gpu,b_gpu,numpy.int32(dim_x), numpy.int32(dim_y), numpy.int32(dim_z),block=(block_size,block_size,1),
grid=((dim_x - 1) / block_size + 1,(dim_y - 1) / block_size + 1))

b=b_gpu.get()

t2=time.time()

tgpu=t2-t1

#Test with CPU 

t3=time.time()
for i in range(0,dim_x):
    for j in range(0,dim_y):
       indice=(nd.convolve1d(a[:,j,i],kernel,mode='reflect')).argmin()      
       b2[j,i]=indice


t4=time.time()
tcpu=t4-t3
print "Difference between CPU:\n"
print b2-b
print ("t_CPU= %f" %tcpu)
#print tcpu
print ("t_GPU= %f" %tgpu)
#print tgpu
print ("Tempo allocazione degli stacks: %f" %talloc)
#print talloc
mem_info=pycuda.driver.mem_get_info()
print "Memoria libera prima di deallocare= %f MB" %(mem_info[0]/1000000)
a_gpu.gpudata.free()
b_gpu.gpudata.free()
aMod_gpu.gpudata.free()
mem_info=pycuda.driver.mem_get_info()
print "Memoria libera= %f MB" %(mem_info[0]/1000000)
print "Memoria totale= %f MB" %(mem_info[1]/1000000)
