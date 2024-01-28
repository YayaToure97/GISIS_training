from modeling import scalar

def simulation():

    id = 0

    myWave = [scalar.Wavefield_1D(), 
              scalar.Wavefield_2D(),
              scalar.Wavefield_3D()] 

    # print(myWave[id]._type)
    myWave[id].get_type()

    myWave[id].set_wavelet()
    myWave[id].plot_wavelet() 

    myWave[id].set_model()
    myWave[id].plot_model()

    myWave[id].set_wave_equation()
    myWave[id].plot_wave_equation()


if __name__ == "__main__":
    simulation()


