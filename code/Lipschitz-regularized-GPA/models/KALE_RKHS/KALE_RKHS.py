#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------
import os
import numpy as np
import re
from functools import partial
import sys

main_dir = os.getcwd()
if "GPA_RKHS" not in main_dir:
    os.chdir("models/GPA_RKHS")
sys.path.append('../../')

# input parameters --------------------------------------
from shared_lib.input_args import input_params
p, _ = input_params()

import yaml
from yaml.loader import SafeLoader
yaml_file = "../../configs/{dataset}-{phi_model}.yaml".format(dataset=p.dataset, phi_model=p.phi_model)

with open(yaml_file, 'r') as f:
    param = yaml.load(f, Loader=SafeLoader)
    #print(param)
    
updated_param = vars(p)
for param_key, param_val in updated_param.items():
    if type(param_val) == type(None):
        continue
    param[param_key] = param_val

if param['alpha']:
    par = [param['alpha']]
    param['exptype'] = '%s=%05.2f-%s' % (param['f'], param['alpha'], param['Gamma'])
else:
    par = []
    param['exptype'] = '%s-%s' % (param['f'], param['Gamma'])
if param['L'] == None:
    param['expname'] = '%s_%s' % (param['exptype'], 'inf')
else:
    param['expname'] = '%s_%.4f' % (param['exptype'], param['L'])

# Data generation ----------------------------------------
from shared_lib.generate_data import generate_data
# X_ ~ Q, Y_ ~ P_0
param, X_, Y_, X_label, Y_label = generate_data(param)


# Discriminator learning  -----------------------------------------
# kernel for RKHS
def gaussian_kernel(x, y, bandwidth):
    return np.exp(-np.sum(((x-y)/bandwidth)**2)/2)
    
def kernel(X_, Y_, bandwidth=1):
    res = np.zeros((X_.shape[0],Y_.shape[0]))
    for i, x in enumerate(X_):
        for j, y in enumerate(Y_):
            res[i,j] = gaussian_kernel(x, y, bandwidth=bandwidth)
    return res
    
k = partial(kernel, bandwidth=param['bandwidth'])


# (Discriminator) Loss ----------------------------------------------
if param['f'] == "KL":
    f = lambda x: x * np.log(x)
    f_prime = lambda x: np.log(x) + 1
    f_2prime = lambda x: 1/x
    f_star = lambda x: np.exp(x - 1)
elif param['f'] == 'alpha':
    f = lambda x: (x**param['alpha'] - 1)/(param['alpha']*(param['alpha']-1))
    f_prime = lambda x: x**(param['alpha']-1)/(param['alpha']-1)
    f_2prime = lambda x: x**(param['alpha']-2)
    f_star = lambda x: 1/param['alpha']*(1/(param['alpha']-1) + ((param['alpha']-1)*np.maximum(x,0))**(param['alpha']/(param['alpha']-1)))
    

def loss(X_, Y_, alpha, lamda):
# loss = \sum_i alpha_i f'(n*alpha_i) - \sum_i f^*(f'(n*alpha_i))/n +
#        1/2* (\sum_i \sum_j k(Y_i, Y_j)/n^2 - 2/n* alpha_i * k(X_i, Y_j) + alpha_i*alpha_j * k(X_i, X_j)
    n = X_.shape[0]
    return np.dot(alpha, f_prime(n*alpha)) -np.sum(f_star(f_prime(n*alpha)))/n + 1/(2*lamda) * (np.sum((k(Y_, Y_)/n**2 + np.dot(k(X_,X_), np.outer(alpha,alpha))).flatten()) - 2/n*np.sum(k(Y_, X_)@alpha))
    
def grad_loss(X_, Y_, alpha, lamda):
    n = X_.shape[0]
    K_XY = np.round(np.linalg.norm(np.sum(k(X_, Y_), axis=1)/(n*lamda)), 4)
    K_XX = np.round(np.linalg.norm(k(X_, X_) @ alpha/(2*lamda)), 4)
    f_prime_n_alpha = np.round(np.linalg.norm(f_prime(n*alpha)), 4)
    #print(r"gradient loss: norm of $K_{XY}, K_{XX}, f'(n \alpha)$ :", K_XY, K_XX, f_prime_n_alpha)
    return f_prime(n*alpha) - np.sum(k(X_, Y_), axis=1)/(n*lamda) + k(X_, X_) @ alpha/(2*lamda)
    
def hess_loss(X_, Y_, alpha, lamda):
    n = X_.shape[0]
    return np.diag(n*f_2prime(n*alpha)) + k(X_, X_)/(2*lamda)
    
def Newton(alpha, lr_phi, grad, hess):
    #print(np.linalg.eigvals(np.linalg.inv(hess_loss(X_, Y_, alpha))))
    return alpha - lr_phi*np.linalg.inv(hess) @ grad
    
def BFGS(alpha, lr_phi, grad, B_inv, X_, Y_, lamda):
    p_k = - B_inv @ grad
    s_k = lr_phi * p_k
    alpha = alpha + s_k
    y_k = grad_loss(X_, Y_, alpha, lamda) - grad
    B_inv += (np.dot(s_k, y_k) + np.sum((B_inv * np.outer(y_k, y_k)).flatten()))/np.dot(s_k, y_k)**2 * np.outer(s_k, s_k) - (B_inv @ np.outer(y_k, s_k.T) + np.outer(s_k, y_k) @ B_inv)/np.dot(s_k, y_k)
    return alpha, B_inv
        

# Transporting particles --------------------------------------------
# gradient of first variation
def grad_gaussian_kernel(x, y, bandwidth):
# gradient with respect to first argument (x) of gaussian kernel k(x,y)
    return - (x-y)/bandwidth**2 * gaussian_kernel(x,y,bandwidth)
    
def grad_kernel(X_, Y_, bandwidth):
    res = np.zeros((X_.shape[0], Y_.shape[1], Y_.shape[0]))
    for i, x in enumerate(X_):
        for j, y in enumerate(Y_):
            res[i,:,j] = grad_gaussian_kernel(x, y, bandwidth)
    return res

grad_k = partial(grad_kernel, bandwidth=param['bandwidth'])

def loss_first_variation(y, X_, Y_, alpha, lamda):
    n = X_.shape[0]
    return 1/lamda * (1/n * np.sum(k(y, Y_) - k(y, X_) @ alpha, axis=1))

def grad_loss_first_variation(y, X_, Y_, alpha, lamda):
    n = X_.shape[0]
    return 1/lamda * (1/n * np.sum(grad_k(y, Y_), axis=2) - grad_k(y, X_) @ alpha)

# ODE solver setting --------------------------------------------------
#from lib.transport_particles import solve_ode
dPs = []
if param['ode_solver'] in ['forward_euler', 'AB2', 'AB3', 'AB4', 'AB5']:
    aux_params = []
else:
    aux_params = {'parameters': parameters, 'phi': phi, 'Q': Q, 'lr_phi': lr_phi,'epochs_phi': param['epochs_phi'], 'loss_par': loss_par, 'NN_par': NN_par, 'data_par': data_par, 'optimizer': param['optimizer']}

# Applying mobility to particles
#if param['mobility'] == 'bounded':
#     bounded_relu = lambda x : np.maximum(x,0)-np.maximum(x-1,0)  # mobility that bounding particles (For image data)
        
# Train setting
lr_P_init = param['lr_P'] # Assume that deltat = deltat(t)
if param['ode_solver'] == "DOPRI5": # deltat = deltat(x,t)
    lr_P_init = [param['lr_P']]*param['N_samples_P']
    # Low dimensional example=> rank 2, Image example=> rank 4
    for i in range(1, Y_.ndim):
        lr_P_init = np.expand_dims(lr_P_init, axis=i)
lr_P = lr_P_init
lr_Ps = []


# Save & plot settings -----------------------------------------------
# Metrics to calculate
from shared_lib.evaluate_metric import calc_ke, calc_grad_phi
if np.prod(param['N_dim']) > 100:
    from shared_lib.evaluate_metric import calc_fid
trajectories = []
vectorfields = []
divergences = []
KE_Ps = []
FIDs = []

# saving/plotting parameters
if param['save_iter'] >= param['epochs']:
    param['save_iter'] = 1

if param['plot_result'] == True:
    from shared_lib.plot_result import plot_result

if not os.path.exists(main_dir + '/assets/' + param['dataset']):
    os.makedirs(main_dir + '/assets/' + param['dataset'])
    
param['expname'] = param['expname']+'_%04d_%04d_%02d_%s' % (param['N_samples_Q'], param['N_samples_P'], param['random_seed'], param['exp_no'])
filename = main_dir + '/assets/' + param['dataset']+'/%s.pickle' % (param['expname'])

if param['plot_intermediate_result'] == True:
    if 'gaussian' in param['dataset'] and 'Extension' not in param['dataset']:
         r_param = param['sigma_Q']
    elif 'student_t' in param['dataset']:
        r_param = param['nu']
    elif param['dataset'] == 'Extension_of_gaussian':
        r_param = param['a']
    else:
        r_param = None
    
# additional plots for simple low dimensional dynamics
if param['N_dim'] == 1:
    xx = np.linspace(-10, 10, 300)
    phis = []
elif param['N_dim'] == 2:#'2D' in param['dataset']:
    xx = np.linspace(-10, 10, 40)
    yy = np.linspace(-10, 10, 40)
    XX, YY = np.meshgrid(xx, yy)
    xx = np.concatenate((np.reshape(XX, (-1,1)), np.reshape(YY, (-1,1))), axis=1)
    phis = []
    
    
# Train ---------------------------------------------------------------
import time 
t0 = time.time()

P, Q = Y_, X_
alpha = np.ones(param['N_samples_P'])
for it in range(1, param['epochs']+1): # Loop for updating particles P
    for in_it in range(param['epochs_phi']):
        grad = grad_loss(Q, P, alpha, param['lamda'])
        if param['optimizer'] == 'Newton':
            hess = hess_loss(Q, P, alpha, param['lamda'])
            alpha = Newton(alpha, param['lr_phi'], grad, hess)
        elif param['optimizer'] == 'BFGS':
            if in_it == 0:
                hess = hess_loss(Q, P, alpha, param['lamda'])
                B_inv = np.linalg.inv(hess)
                alpha, B_inv = BFGS(alpha, param['lr_phi'], grad, B_inv, Q, P, param['lamda'])
    current_loss = loss(X_, Y_, alpha, param['lamda'])
            
    dP = grad_loss_first_variation(P, Q, P, alpha, param['lamda'])
    dPs.append(dP)
    
    P = P - lr_P * dPs[-1]
    '''
    if param['ode_solver'] == "DOPRI5": # deltat adust
        P, dPs, dP, lr_P = solve_ode(P, lr_P, dPs, param['ode_solver'], aux_params) # update P
    else:
        P, dPs, dP = solve_ode(P, lr_P, dPs, param['ode_solver'], aux_params) # update P
    '''

    if param['mobility'] == 'bounded':
        P = bounded_relu(P)
     
    lr_Ps.append(lr_P)
    # adjust learning rates
    #if it>=100:
    #    lr_P = decay_learning_rate(lr_P, param['lr_P_decay'], {'epochs': param['epochs']-100, 'epoch': it-100, 'KE_P': KE_P})
    
    # save results
    divergences.append(current_loss)
    KE_P = calc_ke(dP, param['N_samples_P'])
    KE_Ps.append(KE_P)
    grad_phi = calc_grad_phi(dP)
    #print("grad", grad_phi)
    
    if param['epochs']<=100 or it%param['save_iter'] == 0:
        if param['dataset'] in ['BreastCancer',]:
            trajectories.append(P*10)
        else:
            trajectories.append(P)
        if np.prod(param['N_dim']) < 500:
            vectorfields.append(dP)
        elif np.prod(param['N_dim']) >= 784:  # image data
            FIDs.append( calc_fid(pred=P, real=Q) )
    
    # display intermediate results
    if it % (param['epochs']/10) == 0:
    #if it in [5, 50, 500, 1000, 2000, 3000, 4000, 5000]:
        display_msg = 'iter %6d: loss = %.10f, kinetic energy of P = %.10f, average learning rate for P = %.6f' % (it, current_loss, KE_P, np.mean(lr_P))
        if np.prod(param['N_dim']) >= 784:
            display_msg = display_msg + ', FID = %.3f' % FIDs[-1]
        print(display_msg)
        print("grad", grad_phi)
        
        if param['plot_intermediate_result'] == True:
            data = {'trajectories': trajectories, 'divergences': divergences, 'KE_Ps': KE_Ps, 'FIDs':FIDs, 'X_':X_, 'Y_':Y_, 'X_label':X_label, 'Y_label':Y_label, 'dt': lr_Ps, 'dataset': param['dataset'], 'r_param': r_param, 'vectorfields': vectorfields, 'save_iter':param['save_iter']}
            if param['N_dim'] ==2:
                data.update({'phi': phi, 'W':W, 'b':b, 'NN_par':NN_par})
            plot_result(filename, intermediate=True, epochs = it, iter_nos = None, data = data, show=False)
        
        '''
        if np.prod(param['N_dim']) <= 2:
            zz = phi(xx,None, W,b,NN_par).numpy()
            zz = np.reshape(zz, -1)
            phis.append(zz)
        '''

total_time = time.time() - t0
print(f'total time {total_time:.3f}s')

# Save result ------------------------------------------------------
import pickle
if '1D' in param['dataset']:
    X_ = np.concatenate((X_, np.zeros(shape=X_.shape)), axis=1)
    Y_ = np.concatenate((Y_, np.zeros(shape=Y_.shape)), axis=1)
    
    trajectories = [np.concatenate((x, np.zeros(shape=x.shape)), axis=1) for x in trajectories]
    vectorfields = [np.concatenate((x, np.zeros(shape=x.shape)), axis=1) for x in vectorfields]
        
param.update({'X_': X_, 'Y_': Y_, 'lr_Ps':lr_Ps,})
result = {'trajectories': trajectories, 'vectorfields': vectorfields, 'divergences': divergences, 'KE_Ps': KE_Ps, 'FIDs': FIDs,}

if param['dataset'] in ['BreastCancer',]:
    np.savetxt(main_dir + "/data/gene_expression_example/GPL570/"+param['dataset']+'/output_norm_dataset_dim_%d.csv' % param['N_dim'], trajectories[-1], delimiter=",")
        
# Save trained data
with open(filename,"wb") as fw:
    pickle.dump([param, result] , fw)
print("Results saved at:", filename)

# Plot final result
if param['plot_result'] == True:
    plot_result(filename, intermediate=False, show=False)
