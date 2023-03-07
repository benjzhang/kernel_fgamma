#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------
import os
import numpy as np
import re
from functools import partial

# input parameters --------------------------------------
from util.input_args_RKHS import input_params
p, _ = input_params()
param = vars(p)

if p.alpha:    
    par = [p.alpha]
    p.exptype = '%s=%05.2f-%s' % (p.f, p.alpha, p.Gamma)
else: 
    par = []
    p.exptype = '%s-%s' % (p.f, p.Gamma)
if p.L == None:
    p.expname = '%s_%s' % (p.exptype, 'inf')
else:
    p.expname = '%s_%.4f' % (p.exptype, p.L)

# Data generation ----------------------------------------
from util.generate_data import generate_data
# X_ ~ Q, Y_ ~ P_0
p, X_, Y_, X_label, Y_label = generate_data(p)
       


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
    
k = partial(kernel, bandwidth=p.bandwidth)

# (Discriminator) Loss ----------------------------------------------
if p.f == "KL":
    f = lambda x: x * np.log(x)
    f_star = lambda x: np.exp(x - 1)
    f_star_prime = lambda x: np.exp(x - 1)
    f_star_2prime = lambda x: np.exp(x - 1)
elif p.f == 'alpha':
    f = lambda x: (x**p.alpha - 1)/(p.alpha*(p.alpha-1))
    f_star = lambda x: 1/p.alpha*(1/(p.alpha-1) + ((p.alpha-1)*np.maximum(x,0))**(p.alpha/(p.alpha-1)))
    f_star_prime = lambda x: x**(1/(p.alpha-1)) if x>0 else 0
    f_star_2prime = lambda x: 1/(p.alpha-1)*x**(1/(p.alpha-1)-1)


def loss(X_, Y_, Z_, alpha, lamda):
# loss = \sum_i alpha_i f'(n*alpha_i) - \sum_i f^*(f'(n*alpha_i))/n +
#        1/2* (\sum_i \sum_j k(Y_i, Y_j)/n^2 - 2/n* alpha_i * k(X_i, Y_j) + alpha_i*alpha_j * K(X_i, X_j)
    N_samples_P = Y_.shape[0]
    N_samples_Q = X_.shape[0]
    return np.sum(k(Y_, Z_) @ alpha)/N_samples_P - np.sum(f_star(k(X_, Z_) @ alpha))/N_samples_Q - max((alpha @ k(Z_, Z_) @ alpha -1,0))
    
def grad_loss(X_, Y_, Z_, alpha):
    N_samples_P = Y_.shape[0]
    N_samples_Q = X_.shape[0]
    
    grad_penalty = 0
    if alpha @ k(Z_, Z_) @ alpha > 1:
        grad_penalty = k(Z_, Z_) @ alpha
    return np.sum(k(Y_, Z_)@alpha)/N_samples_P - k(Z_, X_) @ f_star_prime(k(X_, Z_) @ alpha)/N_samples_Q - grad_penalty
    
def hess_loss(X_, Y_, Z_, alpha):
    N_samples_P = Y_.shape[0]
    N_samples_Q = X_.shape[0]
    
    hess_penalty = 0
    if alpha @ k(Z_, Z_) @ alpha > 1:
        hess_penalty =  - k(Z_, Z_)
    
    return k(Z_, X_) @ np.diag(f_star_2prime(k(X_, Z_)@alpha)) @ k(X_, Z_)/N_samples_Q - hess_penalty
 
def Gradient_Aescent(alpha, lr_phi, grad):
    return alpha + lr_phi*grad
    
def Newton(alpha, lr_phi, grad, hess):
    return alpha - lr_phi*np.linalg.inv(hess) @ grad
    
def BFGS(alpha, lr_phi, grad, B_inv, X_, Y_, Z_):
    #print("Eigenvalues of inverse hessian:", np.linalg.eigvals(B_inv))
    p_k = - B_inv @ grad
    s_k = lr_phi * p_k
    alpha = alpha + s_k
    y_k = grad_loss(X_, Y_, Z_, alpha) - grad
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
    
grad_k = partial(grad_kernel, bandwidth=p.bandwidth)
    
def grad_loss_first_variation(y, Z_, alpha):
    return grad_k(y, Z_) @ alpha

# ODE solver setting
#from util.transport_particles import calc_vectorfield, solve_ode
dPs = []
if p.ode_solver in ['forward_euler', 'AB2', 'AB3', 'AB4', 'AB5']:
    aux_params = []
else:
    aux_params = {'parameters': parameters, 'phi': phi, 'Q': Q, 'lr_phi': lr_phi,'epochs_nn': p.epochs_nn, 'loss_par': loss_par, 'NN_par': NN_par, 'data_par': data_par, 'optimizer': p.optimizer}

# Applying mobility to particles
#if p.mobility == 'bounded':
    #from util.construct_NN import bounded_relu  # mobility that bounding particles (For image data)
        
# Train setting
lr_P_init = p.lr_P # Assume that deltat = deltat(t)
if p.ode_solver == "DOPRI5": # deltat = deltat(x,t)
    lr_P_init = [p.lr_P]*p.N_samples_P
    # Low dimensional example=> rank 2, Image example=> rank 4
    for i in range(1, Y_.ndim):
        lr_P_init = np.expand_dims(lr_P_init, axis=i)
lr_P = lr_P_init
lr_Ps = []


# Save & plot settings -----------------------------------------------
# Metrics to calculate
from util.evaluate_metric import calc_ke, calc_grad_phi
if "MNIST" in p.dataset or "CIFAR10" in p.dataset:
    from util.evaluate_metric import calc_fid
trajectories = []
vectorfields = []
divergences = []
KE_Ps = []
FIDs = []

# saving/plotting parameters
if p.save_iter >= p.epochs:
    p.save_iter = 1

if p.plot_result == True:
    from plot_result import plot_result

p.expname = p.expname+'_%04d_%04d_%02d_%s' % (p.N_samples_Q, p.N_samples_P, p.random_seed, p.exp_no)
filename = p.dataset+'/%s.pickle' % (p.expname)

if p.plot_intermediate_result == True:
    if 'gaussian' in p.dataset and 'Extension' not in p.dataset:
         r_param = p.sigma_Q
    elif 'student_t' in p.dataset:
        r_param = p.nu
    elif p.dataset == 'Extension_of_gaussian':
        r_param = p.a
    else:
        r_param = None
    
# additional plots for simple low dimensional dynamics
if p.N_dim == 1:
    xx = np.linspace(-10, 10, 300)
    phis = []
elif p.N_dim == 2:#'2D' in p.dataset:
    xx = np.linspace(-10, 10, 40)
    yy = np.linspace(-10, 10, 40)
    XX, YY = np.meshgrid(xx, yy)
    xx = np.concatenate((np.reshape(XX, (-1,1)), np.reshape(YY, (-1,1))), axis=1)
    phis = []
    
    
# Train ---------------------------------------------------------------
import time 
t0 = time.time()

alpha = np.zeros(int(p.N_samples_Q/2)+int(p.N_samples_P/2))
for it in range(1, p.epochs+1): # Loop for updating particles P
    X_idx_for_Z_ = np.random.choice(p.N_samples_Q, int(p.N_samples_Q/2), replace=False)
    Y_idx_for_Z_ = np.random.choice(p.N_samples_P, int(p.N_samples_P/2), replace=False)
    Z_ = np.vstack((X_[X_idx_for_Z_], Y_[Y_idx_for_Z_]))
    #print(f_star_2prime(k(X_, Z_)@alpha), alpha)
    for in_it in range(p.epochs_phi):
        grad = grad_loss(X_, Y_, Z_, alpha)
        if p.optimizer == 'Gradient_Ascent':
            alpha = Gradient_Aescent(alpha, p.lr_phi, grad)
        elif p.optimizer == 'Newton':
            hess = hess_loss(X_, Y_, Z_, alpha)
            alpha = Newton(alpha, p.lr_phi, grad, hess)
        elif p.optimizer == 'BFGS':
            if in_it == 0:
                #hess = hess_loss(X_, Y_, Z_, alpha)
                #print("Eigenvalues of hessian:", np.linalg.eigvals(hess))
                hess = np.identity(len(alpha))
                B_inv = np.linalg.inv(hess)
                alpha, B_inv = BFGS(alpha, p.lr_phi, grad, B_inv, X_, Y_, Z_)
    current_loss = loss(X_, Y_, Z_, alpha, p.lamda)
            
    dP = grad_loss_first_variation(Y_, Z_, alpha)
    dPs.append(dP)
    
    Y_ = Y_ - lr_P * dPs[-1]
    '''
    if p.ode_solver == "DOPRI5": # deltat adust
        P, dPs, dP, lr_P = solve_ode(P, lr_P, dPs, p.ode_solver, aux_params) # update P
    else:
        P, dPs, dP = solve_ode(P, lr_P, dPs, p.ode_solver, aux_params) # update P
    '''

    if p.mobility == 'bounded':
        Y_ = bounded_relu(Y_)
     
    lr_Ps.append(lr_P)
    # adjust learning rates
    #if it>=100:
    #    lr_P = decay_learning_rate(lr_P, p.lr_P_decay, {'epochs': p.epochs-100, 'epoch': it-100, 'KE_P': KE_P})
    
    # save results
    divergences.append(current_loss)
    KE_P = calc_ke(dP, p.N_samples_P)
    KE_Ps.append(KE_P)
    grad_phi = calc_grad_phi(dP)
    #print("grad", grad_phi)
    
    if p.epochs<=100 or it%p.save_iter == 0:
        if p.dataset in ['BreastCancer',]:
            trajectories.append(Y_*10)
        else:
            trajectories.append(Y_)
        if np.prod(p.N_dim) < 500:
            vectorfields.append(dP)
        elif np.prod(p.N_dim) >= 784:  # image data
            FIDs.append( calc_fid(pred=Y_, real=X_) )
    
    # display intermediate results
    if it % (p.epochs/10) == 0:
    #if it in [5, 50, 500, 1000, 2000, 3000, 4000, 5000]:
        display_msg = 'iter %6d: loss = %.10f, kinetic energy of P = %.10f, average learning rate for P = %.6f' % (it, current_loss, KE_P, np.mean(lr_P))
        if np.prod(p.N_dim) >= 784:
            display_msg = display_msg + ', FID = %.3f' % FIDs[-1]
        print(display_msg)
        print("grad", grad_phi)
        
        if p.plot_intermediate_result == True:
            data = {'trajectories': trajectories, 'divergences': divergences, 'KE_Ps': KE_Ps, 'FIDs':FIDs, 'X_':X_, 'Y_':Y_, 'X_label':X_label, 'Y_label':Y_label, 'dt': lr_Ps, 'dataset': p.dataset, 'r_param': r_param, 'vectorfields': vectorfields, 'save_iter':p.save_iter}
            if p.N_dim ==2:
                data.update({'phi': phi, 'W':W, 'b':b, 'NN_par':NN_par})
            plot_result(filename, intermediate=True, epochs = it, iter_nos = None, data = data, show=False)
        
        '''
        if np.prod(p.N_dim) <= 2:
            zz = phi(xx,None, W,b,NN_par).numpy()
            zz = np.reshape(zz, -1)
            phis.append(zz)
        '''

total_time = time.time() - t0
print(f'total time {total_time:.3f}s')

# Save result ------------------------------------------------------
import pickle
if not os.path.exists(p.dataset):
    os.makedirs(p.dataset)

if '1D' in p.dataset:
    X_ = np.concatenate((X_, np.zeros(shape=X_.shape)), axis=1)
    Y_ = np.concatenate((Y_, np.zeros(shape=Y_.shape)), axis=1)
    
    trajectories = [np.concatenate((x, np.zeros(shape=x.shape)), axis=1) for x in trajectories]
    vectorfields = [np.concatenate((x, np.zeros(shape=x.shape)), axis=1) for x in vectorfields]
        
param.update({'X_': X_, 'Y_': Y_, 'lr_Ps':lr_Ps,})
result = {'trajectories': trajectories, 'vectorfields': vectorfields, 'divergences': divergences, 'KE_Ps': KE_Ps, 'FIDs': FIDs,}

if p.dataset in ['BreastCancer',]:
    np.savetxt("gene_expression_example/GPL570/"+p.dataset+'/output_norm_dataset_dim_%d.csv' % p.N_dim, trajectories[-1], delimiter=",")
        
# Save trained data
with open(filename,"wb") as fw:
    pickle.dump([param, result] , fw)
print("Results saved at:", filename)

# Plot final result
if p.plot_result == True:
    plot_result(filename, intermediate=False, show=False)
