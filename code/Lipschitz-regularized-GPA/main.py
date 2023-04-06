#!/usr/bin/env python3

from sys import argv
import os

if "plot_result" in argv[1]:
    run_script ="python3 shared_lib/plot_result.py {filename}".format(filename=argv[2])
else:
    from shared_lib.input_args import input_params
    p, _ = input_params()

    if 'Latent' in p.phi_model:
        model_dir = p.phi_model[6:]
    else:
        model_dir = p.phi_model
        
    run_script ="python3 models/{model_dir}/{model}.py".format(model_dir = model_dir, model=p.phi_model)

    for a in argv[1:]:
        run_script += " " + a

os.system(run_script)
