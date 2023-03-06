import argparse
    
def input_params():
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument(
        '-N_Q', '--N_samples_Q', type=int, default=200, help='total number of target samples',
    )
    parser.add_argument(
        '-N_P', '--N_samples_P', type=int, default=200, help='total number of prior samples',
    )
    parser.add_argument(
        '-N_dim', type=int, help='dimension of input data',
    )
    parser.add_argument(
        '-N_latent_dim', type=int, help='dimension of latent space',
    )
    parser.add_argument(
        '-N_project_dim', type=int, help='dimension of PCA projected space on input',
    )
    parser.add_argument(
        '-sample_latent', type=bool, default = False, help='True: sample in the latent space, False: sample in the physical space',
    )
    # Dataset property
    parser.add_argument(
        '--dataset', type=str, default='Learning_gaussian', choices=['Learning_gaussian', 'Mixture_of_gaussians', 'Mixture_of_gaussians2','Mixture_of_gaussians3','Mixture_of_gaussians4', 'Stretched_exponential', 'Learning_student_t', 'Mixture_of_student_t', 'Mixture_of_student_t_submnfld', 'Mixture_of_gaussians_submnfld','MNIST', 'CIFAR10', 'MNIST_switch', 'CIFAR10_switch', 'MNIST_ae', 'MNIST_ae_switch','CIFAR10_ae',  'Mixture_of_gaussians_submnfld_ae','BreastCancer', '1D_pts', '2D_pts','1D_dirac2gaussian', '1D_dirac2uniform','Lorenz63']
    )
    parser.add_argument(
        '-beta', type=float, help='gibbs distribution of -|x|^\beta',
    )
    parser.add_argument(
        '-sigma_P', type=float, help='std of initial gaussian distribution',
    )
    parser.add_argument(
        '-sigma_Q', type=float, help='std of target gaussian distribution',
    )
    parser.add_argument(
        '-nu', type=float, help='df of target student-t distribution',
    )
    parser.add_argument(
        '-interval_length', type=float, help='interval length of the uniform distribution',
    )
    parser.add_argument(
        '-label', type=int, nargs="+", help='class label of image data',
    )
    parser.add_argument(
        '-pts_P', type=float, nargs="+", default=[10.0,]
    )
    parser.add_argument(
        '-pts_Q', type=float, nargs="+", default=[0.0,]
    )
    parser.add_argument(
        '-pts_P_2', type=float, nargs="+", default=[0.0,]
    )
    parser.add_argument(
        '-pts_Q_2', type=float, nargs="+", default=[0.0,]
    )
    parser.add_argument(
        '-y0', type=float, nargs="+", default=[1.0,2.0, 2.0]
    )
    parser.add_argument(
        '--random_seed', type=int, default=0, help='random seed for data generator',
    )
    
    
    # (f, Gamma)-divergence
    parser.add_argument(
        '--f', type=str, default='KL', choices=['KL', 'alpha', 'reverse_KL', 'JS'],
    )
    parser.add_argument(
        '-alpha', type=float, help='parameter value for alpha divergence',
    )    
    parser.add_argument(
        '--formulation', type=str, default='LT', choices=['LT', 'DV'], help='LT or DV in case of f=KL, otherwise, keep LT',
    )
    parser.add_argument(
        '--Gamma', type=str, default='Lipshitz', choices=['Lipshitz'],
    )
    parser.add_argument(
        '-L', type=float, help='Lipshitz constant: default=inf w/o constraint',
    )
    parser.add_argument(
        '--reverse', type=bool, default=False, help='True -> D(Q|P), False -> D(P|Q)',
    )
    parser.add_argument(
        '--constraint', type=str, default='hard', choices=['hard', 'soft'],
    )
    parser.add_argument(
        '-lamda', type=float, default=100.0, help='coefficient on soft constraint',
    )
      
    
    # RKHS definition <phi>
    parser.add_argument(
        '--kernel', type=str, default='gaussian', choices=['gaussian'],
    )
    parser.add_argument(
        '-bandwidth', type=float, default = 2.0, help='gaussian kernel bandwidth',
    )
    
    
    # training parameters
    parser.add_argument(
        '-ep', '--epochs', type=int, default=1000, help='# updates for P',
    )
    parser.add_argument(
        '-ep_phi', '--epochs_phi', type=int, default=3, help='# updates to find phi*',
    )
    parser.add_argument(
        '--optimizer', type=str, choices=['Newton', 'BFGS',], default='BFGS', help='optimization method for phi',
    )
    parser.add_argument(
        '--ode_solver', type=str, choices=['forward_euler', 'AB2', 'AB3', 'AB4', 'AB5', 'ABM1', 'Heun', 'ABM2', 'ABM3', 'ABM4', 'ABM5', 'RK4', 'ode45' ], default='forward_euler', help='ode solver for particle ode',
    )
    parser.add_argument(
        '-mobility', type=str, help='problem dependent mobility function\nRecommendation: MNIST - bounded',
    )
    parser.add_argument(
        '-lr_P_decay', type=str, choices=['rational', 'step',], help='delta t decay',
    )
    parser.add_argument(
        '--lr_P', type=float, default=1.0, help='lr for P',
    )
    parser.add_argument(
        '--lr_phi', type=float, default=0.001, help='lr for phi',
    )
    parser.add_argument(
        '--exp_no', type=str, default=0, help='short experiment name under the same data',
    )
    parser.add_argument(
        '--mb_size_P', type=int, default=200, help='mini batch size for the moving distribution P',
    )
    parser.add_argument(
        '--mb_size_Q', type=int, default=200, help='mini batch size for the target distribution Q',
    )
    
    
    # save/display 
    parser.add_argument(
        '--save_iter', type=int, default=10, help='save results per each save_iter',
    )
    parser.add_argument(
        '--plot_result', type=bool, default=False, help='True -> show plots',
    )
    parser.add_argument(
        '--plot_intermediate_result', type=bool, default=False, help='True -> save intermediate plots',
    )
    
    return parser.parse_known_args()
