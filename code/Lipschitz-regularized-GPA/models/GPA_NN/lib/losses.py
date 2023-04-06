import tensorflow as tf
import numpy as np

#@tf.function
def f_star(g, loss_par): # f^*(g) = inf_x <x,g>-f(x)
    threshold = 0.01
    if loss_par['f'] == 'KL': # f = xlogx -x +1
        return tf.math.exp(g)-1
    elif loss_par['f'] == 'alpha':
        alpha = loss_par['par'][0] # f = (x^alpha-1)/(alpha*(alpha-1))
        return 1/alpha*(1/(alpha-1)+tf.math.pow((alpha-1)*tf.nn.relu(g), alpha/(alpha-1)))
    elif loss_par['f'] == 'reverse_KL': # f = -logx
        max_g = tf.math.reduce_max(g)
        if max_g > threshold: # numerical stability
            return -1 - tf.math.log(-g+threshold)
        else:
            return -1 - tf.math.log(-g)
    elif loss_par['f'] == 'JS': # Jensen-Shannon
        max_exp_g = tf.math.reduce_max(tf.math.exp(g))
        if max_exp_g > 2.0-threshold: # numerical stability
            return -tf.math.log( 2 - tf.math.exp(g) + threshold )
        else:
            return -tf.math.log( 2 - tf.math.exp(g) )
            

# --------------------------
def eval_phi(x, x_label, label_idx, N_x, mb_size_x, phi, nu, W, b, NN_par):
    if NN_par['N_conditions'] > 1: # indexing for x_label
        g1 = []
        for idx in label_idx:
            g1.append(tf.math.reduce_mean(phi(x[idx], x_label[idx], W,b, NN_par)))
        return tf.convert_to_tensor(g1, dtype=tf.float32)/NN_par['N_conditions']
            
    else: # not indexing for x_label=None
        g1 = [phi(x[n*mb_size_x:(n+1)*mb_size_x],x_label, W,b, NN_par)/N_x for n in range(int(N_x/mb_size_x))]
        return sum(g1)
    
def eval_fstar_phi(x, x_label, label_idx, N_x, mb_size_x, phi, nu, W, b, NN_par, loss_par):
    if NN_par['N_conditions'] > 1: # indexing for x_label
        g2 = []
        for idx in label_idx:
            if loss_par['formulation'] == 'DV':
                g2.append(tf.math.reduce_mean(tf.math.exp(phi(x[idx], x_label[idx], W,b, NN_par))))
            else: # LT
                g2.append(tf.math.reduce_mean(f_star(phi(x[idx], x_label[idx], W,b, NN_par)-nu, loss_par)))
        return tf.convert_to_tensor(g2, dtype=tf.float32)/NN_par['N_conditions']
        
    else: # don't do indexing for x_label=None
        if loss_par['formulation'] == 'DV':
            g2 = [tf.math.exp(phi(x[n*mb_size_x:(n+1)*mb_size_x],x_label, W,b, NN_par))/N_x for n in range(int(N_x/mb_size_x))]
        else: # LT
            g2 = [f_star(phi(x[n*mb_size_x:(n+1)*mb_size_x],x_label, W,b, NN_par)-nu, loss_par)/N_x for n in range(int(N_x/mb_size_x))]
        return sum(g2)
            


def divergence_mb(phi, nu, P, Q, W, b, NN_par, loss_par, data_par):
    N_P, N_Q, mb_size_P, mb_size_Q = data_par['N_samples_P'], data_par['N_samples_Q'], data_par['mb_size_P'], data_par['mb_size_Q']
    P_label, Q_label = data_par['P_label'], data_par['Q_label']
    label_idx_P, label_idx_Q = data_par['label_idx_P'], data_par['label_idx_Q']
    
    if loss_par['reverse'] == False: # D(P||Q)
        g1 = eval_phi(P, P_label, label_idx_P, N_P, mb_size_P, phi, nu, W, b, NN_par)
        g2 = eval_fstar_phi(Q, Q_label, label_idx_Q, N_Q, mb_size_Q, phi, nu, W, b, NN_par, loss_par)
    else: # D(Q||P)
        g1 = eval_phi(Q, Q_label, label_idx_Q, N_Q, mb_size_Q, phi, nu, W, b, NN_par)
        g2 = eval_fstar_phi(P, P_label, label_idx_P, N_P, mb_size_P, phi, nu, W, b, NN_par, loss_par)
    
    if loss_par['formulation'] == 'DV':
        return tf.reduce_sum(g1) - tf.math.log(tf.reduce_sum(g2))
    else:
        return tf.reduce_sum(g1) - tf.reduce_sum(g2) - nu
        

# @tf.function : might not be compatible to soft lipschitz constraint loss
def divergence_large(phi, nu, P, Q, W, b, NN_par, loss_par, data_par):
    N_P, N_Q, mb_size_P, mb_size_Q = data_par['N_samples_P'], data_par['N_samples_Q'], data_par['mb_size_P'], data_par['mb_size_Q']
    P_label, Q_label = data_par['P_label'], data_par['Q_label']
    
    if loss_par['reverse'] == False: # D(P||Q)
        if NN_par['N_conditions'] > 1: # indexing for P_label
            g1 = [phi(P[n*mb_size_P:(n+1)*mb_size_P],P_label[n*mb_size_P:(n+1)*mb_size_P], W,b, NN_par)/N_P for n in range(int(N_P/mb_size_P))]
        else: # not indexing for P_label=None
            g1 = [phi(P[n*mb_size_P:(n+1)*mb_size_P],P_label, W,b, NN_par)/N_P for n in range(int(N_P/mb_size_P))]
        E_1 = tf.reduce_sum(sum(g1))
            
            
        if loss_par['formulation'] == 'DV':
            if NN_par['N_conditions'] > 1: # indexing for Q_label
                g2 = [tf.math.exp(phi(Q[n*mb_size_Q:(n+1)*mb_size_Q],Q_label[n*mb_size_Q:(n+1)*mb_size_Q], W,b, NN_par))/N_Q for n in range(int(N_Q/mb_size_Q))]
            else: # don't do indexing for Q_label=None
                g2 = [tf.math.exp(phi(Q[n*mb_size_Q:(n+1)*mb_size_Q],Q_label, W,b, NN_par))/N_Q for n in range(int(N_Q/mb_size_Q))]
        else: # LT
            if NN_par['N_conditions'] > 1: # indexing for Q_label
                g2 = [f_star(phi(Q[n*mb_size_Q:(n+1)*mb_size_Q],Q_label[n*mb_size_Q:(n+1)*mb_size_Q], W,b, NN_par)-nu)/N_Q for n in range(int(N_Q/mb_size_Q))]
            else: # not indexing for Q_label=None
                g2 = [f_star(phi(Q[n*mb_size_Q:(n+1)*mb_size_Q],Q_label, W,b, NN_par)-nu)/N_Q for n in range(int(N_Q/mb_size_Q))]
        E_2 = tf.reduce_sum(sum(g2))
            
    else: # D(Q||P)
        if NN_par['N_conditions'] > 1: # indexing for Q_label
            g1 = [phi(Q[n*mb_size_Q:(n+1)*mb_size_Q],Q_label[n*mb_size_Q:(n+1)*mb_size_Q], W,b, NN_par)/N_Q for n in range(int(N_Q/mb_size_Q))]
        else: # not indexing for Q_label=None
            g1 = [phi(Q[n*mb_size_Q:(n+1)*mb_size_Q],Q_label, W,b, NN_par)/N_Q for n in range(int(N_Q/mb_size_Q))]
        E_1 = tf.reduce_sum(sum(g1))
        
        if loss_par['formulation'] == 'DV':
            if NN_par['N_conditions'] > 1: # indexing for P_label
                g2 = [tf.math.exp(phi(P[n*mb_size_P:(n+1)*mb_size_P],P_label[n*mb_size_P:(n+1)*mb_size_P], W,b, NN_par))/N_P for n in range(int(N_P/mb_size_P))]
            else: # don't do indexing for P_label=None
                g2 = [tf.math.exp(phi(P[n*mb_size_P:(n+1)*mb_size_P],P_label, W,b, NN_par))/N_P for n in range(int(N_P/mb_size_P))]
        else: # LT
            if NN_par['N_conditions'] > 1: # indexing for P_label
                g2 = [f_star(phi(P[n*mb_size_P:(n+1)*mb_size_P],P_label[n*mb_size_P:(n+1)*mb_size_P], W,b, NN_par)-nu)/N_P for n in range(int(N_P/mb_size_P))]
            else: # not indexing for P_label=None
                g2 = [f_star(phi(P[n*mb_size_P:(n+1)*mb_size_P],P_label, W,b, NN_par)-nu)/N_P for n in range(int(N_P/mb_size_P))]
        E_2 = tf.reduce_sum(sum(g2))
        
    if loss_par['formulation'] == 'DV':
        return E_1 - tf.math.log(E_2)
    else:
        return E_1 - E_2 - nu


# @tf.function : might not be compatible to soft lipschitz constraint loss
def divergence(phi, nu, P, Q, W, b, NN_par, loss_par, data_par):
    P_label, Q_label = data_par['P_label'], data_par['Q_label']
    if loss_par['reverse'] == False:
        g1, g2 = phi(P,P_label, W,b, NN_par), phi(Q,Q_label, W,b, NN_par)
    else:
        g1, g2 = phi(Q,Q_label, W,b, NN_par), phi(P,P_label, W,b, NN_par)
        
    if loss_par['formulation'] == 'DV':
        return tf.reduce_mean(g1)-tf.math.log(tf.reduce_mean(tf.math.exp(g2)))
    else: # LT
        return tf.reduce_mean(g1)-tf.reduce_mean(f_star(g2-nu, loss_par)+nu)

def calc_grad_phi(dP_dt):
    return np.mean(np.linalg.norm(dP_dt, axis=1))


# @tf.function : might not be compatible to soft lipschitz constraint loss

def gradient_penalty(phi, P, Q, W, b, NN_par, data_par, lamda):
    P_label, Q_label = data_par['P_label'], data_par['Q_label']
    if NN_par['constraint'] == 'soft':
        L = NN_par['L']
        
        '''
        N_tot = min((200, P.shape[0], Q.shape[0]))
        N_P = int(N_tot*P.shape[0]/(P.shape[0]+Q.shape[0]))
        N_Q = int(N_tot*Q.shape[0]/(P.shape[0]+Q.shape[0]))
        r_P = np.random.randint(int(P.shape[0]/N_P))
        r_Q = np.random.randint(int(Q.shape[0]/N_Q))
        R = tf.concat([P[r_P*N_P:(r_P+1)*N_P], Q[r_Q*N_Q:(r_Q+1)*N_Q]], axis=0)
    
        if P_label != None:
            R_label = tf.concat([P_label[r_P*N_P:(r_P+1)*N_P], Q_label[r_Q*N_Q:(r_Q+1)*N_Q]], axis=0)
        else:
            R_label = None
        '''
        
        R = P
        R_label = P_label
  
            
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(R)
            phi_R = phi(R,R_label, W,b,NN_par)
        dR = tape.gradient(phi_R, R)
        
        grad_phi = calc_grad_phi(dR)
        
    
        
        return tf.multiply(-lamda, tf.math.reduce_mean(tf.nn.relu(tf.math.square(dR/L)-1.0)))
    else:
        return tf.constant(0.0, dtype=tf.float32)

