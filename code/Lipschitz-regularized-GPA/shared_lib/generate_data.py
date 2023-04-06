import numpy as np
import tensorflow as tf
import sys

def generate_data(param):
    # Input
    # param: parameters dictionary
    # Outputs
    # X_, Y_, [X_label, Y_label]
    # param
    if param['N_dim'] == None:
        param['N_dim'] = 2
        
    sys.path.append('../')
    
    if param['dataset'] == 'Learning_gaussian':
        from data.Random_samples import generate_gaussian
        param['expname'] = param['expname']+'_%.2f' % param['sigma_Q']
        X_ = generate_gaussian(size=(param['N_samples_Q'], param['N_dim']), m=0.0, std=param['sigma_Q'], random_seed=param['random_seed']) # target
        Y_ = generate_gaussian(size=(param['N_samples_P'], param['N_dim']), m=10.0, std=1.0, random_seed=param['random_seed']+100) # initial
        
    elif param['dataset'] == 'Mixture_of_gaussians':
        from data.Random_samples import generate_gaussian, generate_four_gaussians
        param['expname'] = param['expname']+'_%.2f' % param['sigma_Q']
        X_ = generate_four_gaussians(size=(param['N_samples_Q'], param['N_dim']), dist=4.0, std=param['sigma_Q'], random_seed=param['random_seed']) # target
        Y_ = generate_gaussian(size=(param['N_samples_P'], param['N_dim']), m=4.0, std=1.0, random_seed=param['random_seed']+100) # initial
        
    elif param['dataset'] == 'Mixture_of_gaussians2':
        from data.Random_samples import generate_gaussian, generate_four_gaussians
        param['expname'] = param['expname']+'_%.2f' % param['sigma_Q']
        X_ = generate_four_gaussians(size=(param['N_samples_Q'], param['N_dim']), dist=8.0, std=param['sigma_Q'], random_seed=param['random_seed']) # target
        Y_ = 8.0*(2*np.random.uniform(size=(param['N_samples_P'], param['N_dim'])) - 1)
        #Y_ = generate_gaussian(size=(param['N_samples_P'], param['N_dim']), m=4.0, std=1.0, random_seed=param['random_seed']+100) # initial
        
    elif param['dataset'] == 'Mixture_of_gaussians3':
        from data.Random_samples import generate_two_gaussians
        param['expname'] = param['expname']+'_%.2f' % param['sigma_Q']
        X_ = generate_two_gaussians(size=(param['N_samples_Q'], param['N_dim']), mus=[[-2,0],[2,0]], std=param['sigma_Q'], p = [0.5, 0.5], random_seed=param['random_seed']) # target
        Y_ = generate_two_gaussians(size=(param['N_samples_P'], param['N_dim']), mus=[[-2,2],[2,-2]],  std=param['sigma_Q'],  p = [0.05, 0.95], random_seed=param['random_seed']) # initial
        
    elif param['dataset'] == 'Mixture_of_gaussians4':
        from data.Random_samples import generate_two_gaussians
        param['expname'] = param['expname']+'_%.2f' % param['sigma_Q']
        X_ = generate_two_gaussians(size=(param['N_samples_Q'], param['N_dim']), mus=[[-2,0],[2,0]], std=param['sigma_Q'], p = [0.5, 0.5], random_seed=param['random_seed']) # target
        Y_ = generate_two_gaussians(size=(param['N_samples_P'], param['N_dim']), mus=[[-2,0],[2,0]],  std=param['sigma_Q'],  p = [0.05, 0.95], random_seed=param['random_seed']) # initial
     
    elif param['dataset'] == 'Stretched_exponential':
        from data.Random_samples import generate_stretched_exponential, generate_gaussian
        param['expname'] = param['expname'] +'_%.2f' % param['beta']
        X_ = generate_stretched_exponential(size=(param['N_samples_Q'], param['N_dim']), beta=param['beta'], random_seed=param['random_seed']) # target
        Y_ = generate_gaussian(size=(param['N_samples_P'], param['N_dim']), m=10.0, std=1.0, random_seed=param['random_seed']+100) # initial

    elif param['dataset'] == 'Learning_student_t':
        from data.Random_samples import generate_student_t, generate_gaussian
        param['expname'] = param['expname'] +'_%.2f' % param['nu']
        X_ = generate_student_t(size=(param['N_samples_Q'], param['N_dim']), m=0.0, nu=param['nu'], random_seed=param['random_seed']) # target
        Y_ = generate_gaussian(size=(param['N_samples_P'], param['N_dim']), m=10.0, std=1.0, random_seed=param['random_seed']+100) # initial

    elif param['dataset'] == 'Mixture_of_student_t':
        from data.Random_samples import generate_gaussian, generate_four_student_t
        param['expname'] = param['expname']+'_%.2f' % param['nu']
        X_ = generate_four_student_t(size=(param['N_samples_Q'], param['N_dim']), dist=10.0, nu=param['nu'], random_seed=param['random_seed']) # target
        Y_ = 10.0*(2*np.random.uniform(size=(param['N_samples_P'], param['N_dim'])) - 1)
        #Y_ = generate_gaussian(size=(param['N_samples_P'], param['N_dim']), m=10.0, std=1.0, random_seed=param['random_seed']+100) # initial
        
    elif 'Mixture_of_student_t_submnfld' in param['dataset']:
        from data.Random_samples import generate_gaussian, generate_embedded_four_student_t
        param['expname'] = param['expname']+'_%.2f' % param['nu']
        di=5
        X_ = generate_embedded_four_student_t(N=param['N_samples_Q'], di=di, df=param['N_dim']-2-di,offset=0.0, dist=10.0, nu=param['nu'], random_seed=param['random_seed']) # target
        #m = (5,5,5,5,5, 15,15, 5,5,5,5,5)
        N_dim = param['N_dim']
        if 'ae' in param['dataset']:
            param['expname'] = param['expname']+'_ae%d' % param['N_latent_dim']
            N_dim = param['N_latent_dim']
        Y_ = generate_gaussian(size=(param['N_samples_P'], N_dim), m=15, std=1.0, random_seed=param['random_seed']+100) # initial
        
    elif 'Mixture_of_gaussians_submnfld' in param['dataset']:
        from data.Random_samples import generate_gaussian, generate_embedded_four_gaussians
        param['expname'] = param['expname']+'_%.2f' % param['sigma_Q']
        di=5
        X_ = generate_embedded_four_gaussians(N=param['N_samples_Q'], di=5, df=param['N_dim']-2-di,offset=0.0, dist=4.0, std=param['sigma_Q'], random_seed=param['random_seed']) # target
        N_dim = param['N_dim']
        if 'ae' in param['dataset']:
            param['expname'] = param['expname']+'_ae%d' % param['N_latent_dim']
            if param['sample_latent'] == True:
                N_dim = param['N_latent_dim']
        Y_ = generate_gaussian(size=(param['N_samples_P'], N_dim), m=8, std=0.5, random_seed=param['random_seed']+100) # initial
        
    elif 'MNIST' in param['dataset']:
        from data.Random_samples import generate_logistic
        from data.MNIST import import_mnist, generate_one_hot_encoding
        param['N_dim'] = [28,28,1]
        
        if 'switch' in param['dataset']: # change one label to another
            param['expname'] = param['expname']+'_%02d_%02d' % (param['label'][0], param['label'][1])
            X_ = import_mnist(N=param['N_samples_Q'], label=param['label'][1], normalized=True, random_seed=param['random_seed']) # target
            
        elif param['label'] == None: # put all labels
            if param['N_conditions'] > 1:
                param['expname'] = param['expname']+'_cond'
            else:
                param['expname'] = param['expname']+'_uncond'
            X_, X_label = import_mnist(N=param['N_samples_Q'], label=param['label'], normalized=True, random_seed=param['random_seed']) # target
            if param['N_samples_P'] == param['N_samples_Q']:
                Y_label = X_label
            else:
                prop = [np.sum(X_label[:, j], keepdims=False)/param['N_samples_Q'] for j in range(np.shape(X_label)[1])]
                data = np.random.choice(np.shape(X_label)[1], param['N_samples_P'], p = prop)
                Y_label = generate_one_hot_encoding(param['N_samples_P'], np.shape(X_label)[1], param['random_seed'], data)
            param['Y_label'] = Y_label
            param['X_label'] = X_label
        else: # random distribution to one designated label
            param['expname'] = param['expname']+'_%02d' % param['label'][0]
            X_ = import_mnist(N=param['N_samples_Q'], label=param['label'], normalized=True, random_seed=param['random_seed']) # target
        N_dim = param['N_dim']
        if 'ae' in param['dataset']:  # transport in a latent space
            param['expname'] = param['expname']+'_ae%d' % param['N_latent_dim']
            if param['sample_latent'] == True:
                if type(param['N_latent_dim']) != list:
                    N_dim = [param['N_latent_dim']]
                else:
                    N_dim = param['N_latent_dim']
        if 'switch' in param['dataset']: # change one label to another
            Y_ = import_mnist(N=param['N_samples_Q'], label=param['label'][0], normalized=True, random_seed=param['random_seed']) # initial
        else:
            if param['N_project_dim'] == None:
                Y_ = generate_logistic(size=tuple([param['N_samples_P']]+N_dim), loc=0.0, scale=1.0, random_seed=param['random_seed']+100) # initial
            else:
                Y_ = projected_logistic(size=tuple([param['N_samples_P']]+N_dim), loc=0.0, scale=1.0, random_seed=param['random_seed']+100, proj_dim=param['N_project_dim']) # initial
    elif 'CIFAR10' in param['dataset']:
        from data.Random_samples import generate_logistic
        from data.CIFAR10 import import_cifar10, generate_one_hot_encoding
        param['N_dim'] = [32,32,3]
        
        if 'switch' in param['dataset']: # change one label to another
            param['expname'] = param['expname']+'_%02d_%02d' % (param['label'][0], param['label[1]'])
            X_ = import_cifar10(N=param['N_samples_Q'], label=param['label'][1], normalized=True, random_seed=param['random_seed']) # target
        elif param['label'] == None: # put all labels
            if param['N_conditions'] > 1:
                param['expname'] = param['expname']+'_cond'
            else:
                param['expname'] = param['expname']+'_uncond'
            X_, X_label = import_cifar10(N=param['N_samples_Q'], label=param['label'], normalized=True, random_seed=param['random_seed']) # target
            if param['N_samples_P'] == param['N_samples_Q']:
                Y_label = X_label
            else:
                prop = [np.sum(X_label[:, j], keepdims=False)/param['N_samples_Q'] for j in range(np.shape(X_label)[1])]
                data = np.random.choice(np.shape(X_label)[1], param['N_samples_P'], p = prop)
                Y_label = generate_one_hot_encoding(param['N_samples_P'], np.shape(X_label)[1], param['random_seed'], data)
            param['Y_label'] = Y_label
            param['X_label'] = X_label
        else: # random distribution to one designated label
            param['expname'] = param['expname']+'_%02d' % param['label'][0]
            X_ = import_cifar10(N=param['N_samples_Q'], label=param['label'], normalized=True, random_seed=param['random_seed']) # target
        
        N_dim = param['N_dim']
        if 'ae' in param['dataset']:  # transport in a latent space
            param['expname'] = param['expname']+'_ae%d' % param['N_latent_dim']
            if param['sample_latent'] == True:
                if type(param['N_latent_dim']) != list:
                    N_dim = [param['N_latent_dim']]
                else:
                    N_dim = param['N_latent_dim']
        if 'switch' in param['dataset']: # change one label to another
            Y_ = import_cifar10(N=param['N_samples_Q'], label=param['label'][0], normalized=True, random_seed=param['random_seed']) # initial
        else:
            if param['N_project_dim'] == None:
                Y_ = generate_logistic(size=tuple([param['N_samples_P']]+N_dim), loc=0.0, scale=1.0, random_seed=param['random_seed']+100) # initial
            else:
                Y_ = projected_logistic(size=tuple([param['N_samples_P']]+N_dim), loc=0.0, scale=1.0, random_seed=param['random_seed']+100, proj_dim=param['N_project_dim']) # initial

    elif param['dataset'] in ['BreastCancer',] :
        from numpy import genfromtxt
        param['expname'] = param['expname']+'_dim%d' % (param['N_dim'] )
        X_ = genfromtxt('../data/gene_expression_example/GPL570/%s/target_norm_dataset_dim_%d.csv' % (param['dataset'], param['N_dim']), delimiter=',')
        Y_ = genfromtxt('../data/gene_expression_example/GPL570/%s/source_norm_dataset_dim_%d.csv' % (param['dataset'], param['N_dim']), delimiter=',')
        
        param['N_samples_Q'], param['N_samples_P'] = len(X_), len(Y_)
    
    elif param['dataset'] == '1D_pts':
        param['N_dim'] = 1
        param['expname'] = param['expname']+'_P%s_Q%s' % ( '-'.join([str(x) for x in param['pts_P']]), '-'.join([str(x) for x in param['pts_Q']]) )
        X_ = np.array(np.reshape(param['pts_Q'], (-1,1)))
        Y_ = np.array(np.reshape(param['pts_P'], (-1,1)))
        
        param['N_samples_Q'] = len(param['pts_Q'])
        param['N_samples_P'] = len(param['pts_P'])
        
    elif param['dataset'] == '2D_pts':
        param['N_dim'] = 2
        X_ = np.empty((len(param['pts_Q']),2))
        Y_ = np.empty((len(param['pts_P']),2))
        list_pts_Q, list_pts_P = [], []
        for i, (x, y) in enumerate(zip(param['pts_Q'], param['pts_Q_2'])):
            X_[i, 0] = x
            X_[i, 1] = y
            list_pts_Q.append(str(x)+","+str(y))
        for i, (x, y) in enumerate(zip(param['pts_P'], param['pts_P_2'])):
            Y_[i, 0] = x
            Y_[i, 1] = y
            list_pts_P.append(str(x)+","+str(y))
        param['expname'] = param['expname']+'_P%s_Q%s' % ( '-'.join([str(x) for x in list_pts_P]), '-'.join([str(x) for x in list_pts_Q]) )
        
        param['N_samples_Q'] = len(param['pts_Q'])
        param['N_samples_P'] = len(param['pts_P'])
        
    elif param['dataset'] == '1D_dirac2gaussian':
        from data.Random_samples import generate_gaussian
        param['N_dim'] = 1
        param['expname'] = param['expname']+'_%.2f' % param['sigma_Q']
        X_ = generate_gaussian(size=(param['N_samples_Q'], param['N_dim']), m=0.0, std=param['sigma_Q'], random_seed=param['random_seed']) # target
        if param['sigma_P']:
            Y_ = generate_gaussian(size=(param['N_samples_P'], param['N_dim']), m=0.0, std=param['sigma_P'], random_seed=param['random_seed']+100) # initial
        else:
            Y_ = generate_gaussian(size=(param['N_samples_P'], param['N_dim']), m=0.0, std=0.01, random_seed=param['random_seed']+100) # initial
            
    elif param['dataset'] == '1D_dirac2uniform':
        from data.Random_samples import generate_gaussian, generate_uniform
        param['N_dim'] = 1
        param['expname'] = param['expname']+'_%.2f' % param['interval_length']
        X_ = generate_uniform(size=(param['N_samples_Q'], param['N_dim']), shift=0.0, l=param['interval_length'], random_seed=param['random_seed']) # target
        if param['sigma_P']:
            Y_ = generate_gaussian(size=(param['N_samples_P'], param['N_dim']), m=0.0, std=param['sigma_P'], random_seed=param['random_seed']+100) # initial
        else:
            Y_ = generate_gaussian(size=(param['N_samples_P'], param['N_dim']), m=0.0, std=0.01, random_seed=param['random_seed']+100) # initial
            
    elif param['dataset'] == 'Lorenz63':
        from data.Lorenz63 import generate_invariant_lorenz63
        from data.Random_samples import generate_gaussian
        param['N_dim'] = 3
        param['expname'] = param['expname']+'_%.1f-%.1f_%.1f' % (param['y0'][0], param['y0'][1], param['y0'][2])
        
        X_ = generate_invariant_lorenz63(size=(param['N_samples_Q'], param['N_dim']), y0=param['y0'], random_seed=param['random_seed']) # target
        Y_ = generate_gaussian(size=(param['N_samples_P'], param['N_dim']), m=0.0, std=0.01, random_seed=param['random_seed']+100) # initial
        
    
    if param['mb_size_P'] > param['N_samples_P']:
        param['mb_size_P'] = param['N_samples_P']
    if param['mb_size_Q'] > param['N_samples_Q']:
        param['mb_size_Q'] = param['N_samples_Q']
    
    if 'X_label' not in locals():
        X_label, Y_label = None, None
    return param, X_, Y_, X_label, Y_label
