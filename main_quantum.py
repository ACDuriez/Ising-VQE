import os
import logging

from solver import Solver

from utils.utils import get_date_postfix
from utils.args import get_VQE_config


def main(config):
    
    # Timestamp
    simlulation_time = get_date_postfix()
    config.saving_dir = os.path.join(config.saving_dir, simlulation_time)
    config.log_dir_path = os.path.join(config.saving_dir, 'log_dir')
    config.data_dir_path = os.path.join(config.saving_dir, 'data_dir')
    config.img_dir_path = os.path.join(config.saving_dir, 'img_dir')

    
    # Create directories if not exist
    if not os.path.exists(config.log_dir_path):
        os.makedirs(config.log_dir_path)
    if not os.path.exists(config.data_dir_path):
        os.makedirs(config.data_dir_path)
    if not os.path.exists(config.img_dir_path):
        os.makedirs(config.img_dir_path)

    
    # Logger
    log_p_name = os.path.join(config.log_dir_path, simlulation_time+ '_logger.log')
    from importlib import reload
    reload(logging)
    logging.basicConfig(filename=log_p_name, level=logging.INFO,format='%(message)s \n\n')
    logger = logging.getLogger(log_p_name)
    logger.info(config)
    config.logger = logger
    #  Logger Optimizer
    log_opt_p_name = os.path.join(config.log_dir_path, simlulation_time+ '_logger_opt.log')
    logging.basicConfig(filename=log_opt_p_name, level=logging.INFO,format='%(message)s \n\n')
    logger_opt = logging.getLogger(log_opt_p_name)
    config.logger_opt = logger_opt
    
    
    #Running simulation
    solver = Solver(config)

    solver.run()

if __name__ == "__main__":

    config = get_VQE_config()

    # Physical system configuration
    # Number of lattice sites
    config.num_sites = 2
    # Phase transition
    config.transition = 'magkink'
    # Transverse field magnitude
    config.hx = 0.5
    # Initial field value
    config.hi = 1.2
    # Final field value
    config.hf = 1.
    # Values of the varying field
    config.field_range = (1.5,1.4) # Decreasing order is preferable
    # Number of values of varying field
    config.field_steps = 2


    # Optimization configurations
    # Previous data points
    config.data_points = []
    # First and last points to optimize
    config.first_point = 0
    config.last_point = None
    # Optimizer 
    config.opt_algorithm = 'L_BFGS_B'
    # Optimizer iterations
    config.opt_iterations = 150
    config.learning_rate = None
    config.learning_rate_step = None
    config.learning_rate_factor = None
    config.perturbation_magnitude = 1e-2
    # Parameter initialization
    config.sampling = 'previous'
    config.param_init_range = 0.1


    # Quantum circuit configurations
    config.ansatz = 'hva'
    config.layers = 2

    # Runtime configurations
    # Enable Runtime
    config.backend_name = 'fake_backend'
    config.server = 'local'
    # Save account credentials
    config.save_account = False

    # Saving directory
    config.saving_dir = 'results/VQE'

    print(config)

    main(config)