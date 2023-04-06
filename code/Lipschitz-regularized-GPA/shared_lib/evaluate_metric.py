import numpy as np
import tensorflow as tf
from skimage.transform import resize
from scipy.linalg import sqrtm
from sys import platform
import torch
from torch.nn.functional import adaptive_avg_pool2d
        
#Code modified from https://github.com/biweidai/SINF/blob/master/sinf/fid_score.py
# Frechet Inception Distance calculation
def load_inception_model_and_preprocess_data(library, data1, data2):
    if library == "tensorflow":
        from tensorflow.keras.applications.inception_v3 import InceptionV3
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        
        input_shape = (max((299,data1.shape[1])),max((299,data1.shape[2])),3)
        try:
            model = InceptionV3(weights='imagenet', input_shape=input_shape, include_top=False, pooling='avg')
        except: # For Mac
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            model = InceptionV3(weights='imagenet', input_shape=input_shape, include_top=False, pooling='avg')
        
        data1 = preprocess_input(data1)
        data2 = preprocess_input(data2)
        
        device = None
            
    elif "torch" in library:
        try:
            from shared_lib.inception import InceptionV3
        except:
            from inception import InceptionV3
        if 'darwin' in platform:
            gpu = "mps"
        elif 'linux' in platform:
            gpu = "cuda"
        else:
            gpu = "cpu"
        device = torch.device(gpu)
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]

        model = InceptionV3([block_idx]).to(device)
        
        data1 = (data1 - np.min(data1)) / (np.max(data1) - np.min(data1)) # values lie in [0,1]
        data2 = (data2 - np.min(data2)) / (np.max(data2) - np.min(data2)) # values lie in [0,1]
        data1 = np.transpose(data1, axes=(0,3,1,2))
        data2 = np.transpose(data2, axes=(0,3,1,2))
        
    return model, data1, data2, device
    
def gray2rgb_images(samples):
    samples = np.concatenate([samples, samples, samples], axis=-1)
    return samples
    
def resize_images(samples, new_shape):
    new_samples = []
    for i in range(samples.shape[0]):
        new_samples.append( resize(samples[i], new_shape, 0) )
    new_samples = np.stack(new_samples, axis=0)
    return new_samples
        
def predict_divide_and_conquer(data, model, library, device, batch_size=5000):
    predicts = []
    n_batches = int(np.ceil(data.shape[0]/batch_size))
    for i in range(n_batches):
        if i < n_batches-1:
            mini_batch = data[i*batch_size: (i+1)*batch_size]
        else:
            mini_batch = data[i*batch_size:len(data)]
            
        if data.shape[-1] < 3: # gray-scale image
            mini_batch = gray2rgb_images(mini_batch)
        if (data.shape[1] < 299) or (data.shape[2] < 299):
            d_shape = (299,299,3) if library == "tensorflow" else (3,299,299)
            mini_batch = resize_images(mini_batch, d_shape)
            
        if library == "tensorflow":
            predicts.append(model.predict(mini_batch))
        elif "torch" in library:
            model.eval()
            mini_batch = torch.FloatTensor(mini_batch)
            mini_batch = mini_batch.to(device)
            with torch.no_grad():
                pred = model(mini_batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            predicts.append( pred.squeeze(3).squeeze(2).cpu().numpy() )
   
    predict = np.concatenate(predicts, axis=0)
    return predict
    
    
def calc_fid(pred, real, batch_size=500):
# Frechet Inception Distance calculated on "dataset" using Inception V3
# pred: model prediction image(s) of 3D or 4D numpy array
# real: real image(s) of 3D or 4D numpy array
    library = "pytorch"#"tensorflow"
    if "linux" in platform:
        batch_size = 20
    model, pred, real, device = load_inception_model_and_preprocess_data(library, pred, real)

    # calculate means, covariances
    mu, sigma = [],[]
    for data in (pred, real):
        predict = predict_divide_and_conquer(data, model, library, device, batch_size)
        
        # flatten the array of prediction into the shape (N_Features , N_Samples)
        if predict.ndim > 2:  # cnn prediction
            predict = np.reshape(predict, (-1, data.shape[0]), order='C')
        else:  # fnn prediction
            predict = np.transpose(predict)

        mu.append(np.mean(predict, axis=1))
        sigma.append(np.cov(predict))
        
    # calculate frechet distance
    diff = mu[0]-mu[1]
    covmean = sqrtm(sigma[0] @ sigma[1])
    if np.iscomplex(covmean).any():
        #print("Convert imaginary numbers entries to real numbers")
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
        
    return np.dot(diff, diff) + np.trace(sigma[0]) + np.trace(sigma[1]) - 2 * tr_covmean
    

# ------------------------------------------------------------------------------
def calc_ke(dP_dt, N_samples_P):
    return np.linalg.norm(dP_dt)**2/N_samples_P/2
    
def calc_grad_phi(dP_dt):
    return np.mean(np.linalg.norm(dP_dt, axis=1))

def calc_ke_gan(dD_dSamples, dG_dDiscParams):
# calculate the expected kinetic energy of generated particles in GAN - Sobolev descent: Mroueh, Youssef et al. “Sobolev Descent.” AISTATS (2019).
# dY_dt = 1/n \sum_n \frac{\partial G(z)^t}{\partial \theta } \frac{\partial G(\tilde{z})}{\partial \theta }|_{theta=theta_N} \nabla D_N (G_{\theta_N} (\tilde{z}_n))
# dD_dSamples: numpy array of size (N_samples, 1, Y_dim) - gradient of discriminator with respect to the generated sample
# dG_dDiscParams: numpy array of size (N_samples+1, theta_dim, Y_dim) - gradient of generator with respect to the discriminator parameter 
    N_samples = len(dD_dSamples)
    if len(dD_dSamples.shape) == 2:
        dD_dSamples = np.expand_dims(dD_dSamples, axis=1)
        
    dG_dDiscParams1 = np.tile(dG_dDiscParams[-1], (N_samples,1,1))
    dG_dDiscParams = dG_dDiscParams[:-1]
    
    dY_dt = np.matmul( np.matmul(dD_dSamples, np.transpose(dG_dDiscParams, (0,2,1))), dG_dDiscParams1)
    
    return calc_ke(dY_dt, N_samples)
    
    
def calc_sinkhorn(P, Q, reg=0.2):
    import ot
    N_P = P.shape[0]
    N_Q = Q.shape[0]
    
    # squred Euclidean distance
    M = np.tile(np.sum(P**2, axis=1), (N_Q,1)).T + np.tile(np.sum(Q**2, axis=1), (N_P,1)) - 2*P@Q.T
    gamma = ot.sinkhorn(1/N_P * np.ones(N_P), 1/N_Q * np.ones(N_Q), M, reg = 0.3)
    
    return sum(sum(gamma * M))
   
if __name__ == "__main__":
    # test
    import sys
    if sys.argv[1] == "calc_fid":
        N_samples_pred = int(sys.argv[2])
        N_samples_real = int(sys.argv[3])
        
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
        x_train = x_train/255.0
        
        idx_pred = np.random.permutation(x_train.shape[0])[:N_samples_pred]
        idx_real = np.random.permutation(x_train.shape[0])[:N_samples_real]

        pred = np.expand_dims(x_train[idx_pred], axis=3)
        real = np.expand_dims(x_train[idx_real], axis=3)
        
        print(calc_fid(pred, real))
    elif sys.argv[1] == "calc_ke":
        N_samples_P = sys.arvg[2]
        
        dP = np.random.random((N_samples_P, 10))
        print(calc_ke(dP, N_samples_P))
    elif sys.argv[1] == "calc_ke_gan":
        dD_dSamples = np.random.random((50, 12))
        dG_dDiscParams = np.random.random((51, 1200, 12))
        print(calc_ke_gan(dD_dSamples, dG_dDiscParams))
        
    elif sys.argv[1] == "calc_sinkhorn":
        P = np.random.random((30,1))
        Q = np.random.random((40,1))
        R = np.random.normal(size=(40,1))
        print(calc_sinkhorn(P,Q, reg=sys.argv[2]))
        print(calc_sinkhorn(P,R, reg=sys.argv[2]))
