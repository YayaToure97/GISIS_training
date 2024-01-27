import numpy as np
import matplotlib.pyplot as plt

class Wavefield_1D():
    
    def __init__(self):
        
        self._type = "1D wave propagation in constant density acoustic isotropic media"

        # TODO: read parameters from a file

        self.nt = 1001
        self.dt = 1e-3
        self.fmax = 30.0
        self.nz=1001 # Número total de pontos
        self.dz=10   # Espaçamentos entre os pontos
        
        self.interfaces= np.array([1000,1500,2000,2500]) # definições das interfaces
        self.velocidades= np.array([1500,2000,2500,3000,3500]) # definições das velocidades

        self.depth = np.linspace(0,self.dz*self.nz,self.nz,endpoint=False) # definição da profundidade 
        self.model =   self.velocidades[0]*np.zeros(self.nz) ### criação de um modelo com  matriz vazia
        

        self.fonte= [0,500,1000,2000] # posições das respectivas fontes
        self.receptor=[1000,1500,2500,3000] # posições das respectivas receptores
        
    def set_model(self):
        pass 

        for i in range(len(self.interfaces)):
            self.model[int(self.interfaces[i] / self.dz):] = self.velocidades[i+1] 
    
    def plot_model(self):
        pass 

        fig,ax = plt.subplots(num = 'model plot', figsize=(4,8),clear=True)
        
        ax.plot(self.model,self.depth)
        ax.scatter(self.fonte,self.interfaces,color='black', marker='*', label='Fonte')
        ax.scatter(self.receptor,self.interfaces,color='red', marker='v', label='Receptor')
        ax.set_title('model plot',fontsize=18)
        ax.set_xlabel('Velocity [m/s]',fontsize=15)
        ax.set_ylabel('depth [m]',fontsize =15) 

        ax.set_ylim([0,(self.nz-1)*self.dz])

        ax.invert_yaxis()
        fig.tight_layout()
        plt.show()
            

    def get_type(self):
         print(self._type)

    def set_wavelet(self):
    
         t0 = 2.0*np.pi/self.fmax
         fc = self.fmax/(3.0*np.sqrt(np.pi))
         td = np.arange(self.nt)*self.dt - t0

         arg = np.pi*(np.pi*fc*td)**2.0

         self.wavelet = (1.0 - 2.0*arg)*np.exp(-arg)

    def plot_wavelet(self):
       
         t = np.arange(self.nt)*self.dt

         fig, ax = plt.subplots(figsize = (10, 5), clear = True)

         ax.plot(t, self.wavelet)
         ax.set_title("Wavelet", fontsize = 18)
         ax.set_xlabel("Time [s]", fontsize = 15)
         ax.set_ylabel("Amplitude", fontsize = 15) 
        
         ax.set_xlim([0, np.max(t)])
        
         fig.tight_layout()
         plt.show() 


    


class Wavefield_2D():
    
    def __init__(self):
        super().__init__()
        
        self._type = "2D wave propagation in constant density acoustic isotropic media"    

# Read the parameters from file
# self.nx=200
# self.nz=50

# self.model=np.zeros(nz)
# self.depth=np.arange(nz)*self.nx

# self.deep =[1500,1800,2000,2200,2500]
# self.Density =[1000,2100,2150,2180,2200]
# self.interfaces =[100,150,200,300] 

# Vp=np.zeros((nx,nz))
# Rho=np.zeros((nx,nz)) 


# for i in range(len(interfaces)):
#     vp[int(interfaces[i] / dh):] = velocidades[i+1]
#     rho[int(interfaces[i] / dh):] = densidades[i+1]

# plt.imshow(self.vp*self.rho)
# yaya=plt.colorbar(orientation ='horizontal')
# plt.xlabel('Distância')
# plt.ylabel('Profundidade')
# yaya.set_label('impedância')
# plt.show()


class Wavefield_3D(Wavefield_2D):
    
    def __init__(self):
        super().__init__()
        
        self._type = "3D wave propagation in constant density acoustic isotropic media"    
        
class Wavefield_4D(Wavefield_2D):
    def __init__(self):
        super().__init__()
        
        self.type="3D wave propagation in constant density acoustic isotropic media"

