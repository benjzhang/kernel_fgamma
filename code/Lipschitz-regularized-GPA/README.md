# Particles transportation based on Lipschitz regularized f-divergences
## Goal
Explore the ways to parametrize a discriminator function for training Generative Particles (GPA)
```
python3 main.py --dataset Learning_gaussian --phi_model GPA_NN --exp_no gpa_nn
python3 main.py --dataset Learning_gaussian --phi_model GPA_RKHS --exp_no gpa_rkhs
python3 main.py --dataset Learning_gaussian --phi_model KALE_RKHS --exp_no kale
```
The entire parameter set can be found in configs/Learning_gaussian-{model_name}.yaml.
To make changes to the parameters, see shared_lib/input_args.py and reassign parameter values.
ex) python3 main.py --dataset Learning_gaussian --phi_model GPA_NN --exp_no gpa_nn --epochs 100
    in order to reassign epochs.
