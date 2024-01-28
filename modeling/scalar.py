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
        self.dz=5   # Espaçamentos entre os pontos
        
        self.interfaces= np.array([500,1000,1500,2000]) # definições das interfaces
        self.velocidades= np.array([1000,1500,2000,2500,3500]) # definições das velocidades

        self.depth = np.linspace(0,self.dz*self.nz,self.nz,endpoint=False) # definição da profundidade 
        self.model =   self.velocidades[0]*np.zeros(self.nz) ### criação de um modelo com  matriz vazia
        

        self.fonte= np.array([100,200,300,400]) # posições das respectivas fontes
        self.receptor=np.array([500,1000,1500,2500]) # posições das respectivas receptores
        
    def set_model(self):
        pass 

        for i in range(len(self.interfaces)):
            self.model[int(self.interfaces[i] / self.dz):] = self.velocidades[i+1] 
    
    def plot_model(self):
        pass 
    def set_wave_equation(self): # Equação da onda utilizando a solução análitica


        self.nt = 101  # grid in space
        self.nx = 51 # grid in time
        self.a = 0
        self.b = 1
        self.t0 = 0
        self.tf = 0.0005
        self.dx = (self.b - self.a) / (self.nx - 1)
        self.dt = (self.tf - self.t0) / (self.nt - 1)
        self.x = np.linspace(self.a, self.nx*self.dx, self.nx, endpoint=True)  # Corrected
        self.t = np.linspace(self.t0, self.nt*self.dt, self.nt, endpoint=True)  # Corrected

    def plot_wave_equation(self):
        pass 
        self.s = self.dt / self.dx**2
        self.UA = np.zeros((self.nx, self.nt))  # Corrected

        for i in range(self.nx):
            for j in range(self.nt):
                self.UA[i, j] = np.sin(np.pi * self.x[i]) * np.exp(-np.pi**2 * self.t[j])  # Corrected



        fig,ax = plt.subplots(num = 'wave_equation', figsize=(3,8),clear=True)

        ax.contourf(self.UA,200,cmap='viridis')
        ax.set_title('Analytical solution',fontsize=18)
        ax.set_xlabel('X [m]',fontsize=15)
        ax.set_ylabel('Y [s]',fontsize =15) 
        plt.show()
                
        


        fig,ax = plt.subplots(num = 'model plot', figsize=(3,8),clear=True)

        fonte_projecao=np.array(self.fonte/self.dz, dtype=int)
        receptor_projecao=np.array(self.receptor/self.dz, dtype=int)
        
        ax.plot(self.model,self.depth)
        ax.scatter(self.model[fonte_projecao],self.fonte,color='black', marker='*', label='Fonte')
        ax.scatter(self.model[receptor_projecao],self.receptor,color='red', marker='v', label='Receptor')
        ax.set_title('model',fontsize=18)
        ax.set_xlabel('Velocity [m/s]',fontsize=15)
        ax.set_ylabel('depth [m]',fontsize =15) 

        ax.set_ylim([0,(self.nz-1)*self.dz])

        ax.invert_yaxis()
        fig.tight_layout()
        ax.grid(True)
        plt.show()
        plt.savefig('modelo_velocidade _1D.png') 

   

    
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
         plt.savefig('Wavelet.png')
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

