import os
import matlab.engine
import numpy as np
from matplotlib import pyplot as plt
import constants as c


if __name__ == '__main__':
    np.random.seed(42)

    blurred_path = os.path.join(c.DATA_DIR, 'blurred')
    images_path = [os.path.join(blurred_path, f) for f in os.listdir(blurred_path)]
    print('Starting MATLAB engine...')
    eng = matlab.engine.start_matlab()
    eng.addpath(os.path.join(c.PROJECT_DIR, 'matlab'))
    for i, filename in enumerate(images_path):
        condition = int(filename.split('/')[-1].split('_')[3])
        print('Processing', filename)
        eng.deblur_ERCO(filename, nargout=0)
        y_hat = np.array(eng.workspace['y_hat'])
        plt.imsave(os.path.join(c.DATA_DIR, 'deblurred_erco', 'dnn', filename.split('/')[-1]), y_hat, cmap='gray')

    eng.quit()
