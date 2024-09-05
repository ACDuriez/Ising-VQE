import argparse

def str2bool(v):
    return v.lower() in ['true']

def get_VQE_config():
    parser = argparse.ArgumentParser()

    # Quantum Model configurations
    parser.add_argument('--transition', type=str, default='magkink', help='which kind of phase transition, can be: magkink wetting_1st weeting_2nd')
    parser.add_argument('--num_sites', type=int, default='4', help='number of lattice sites')
    parser.add_argument('--hx', type=float, default='0.5', help='transverse field magnitude')
    parser.add_argument('--hi', type=float, default='1.2', help='initial value of varying field')
    parser.add_argument('--hf', type=float, default='1.0', help='final value of varying field')
    parser.add_argument('--aux_observables', type=list, default=[], help='physical observables to measure on groundstate (SparsePauliOp)')

    # Quantum circuit configuration
    parser.add_argument('--ansatz', type=str, default='hva', help='which time of ansatz, hardware efficient (hea) or hamiltonian variational (HVA)')
    parser.add_argument('--layers', type=int, default=1, help='Circuit layers of the Ansatz')

    
    #Simulation configurations
    parser.add_argument('--field_steps', type=int, default=10, help='number of field values to evaluate')
    parser.add_argument('--field_range', type=tuple, default=(0.3,0.8), help='values of the varying magnetic field')
    parser.add_argument('--data_points', type=list, default=[], help='data points of past optimizations')

    # Optimization configurations
    parser.add_argument('--opt_algorithm', type=str, default='L_BFGS_B', help='optimization algorithm')
    parser.add_argument('--opt_iterations', type=int, default=100, help='maximum optimizer iterations')
    parser.add_argument('--learning_rate', type=float, default=0.5, help='learning rate for gradient based optimizers')
    parser.add_argument('--learning_rate_step', type=int, default=20, help='step to decrease learning rate')
    parser.add_argument('--learning_rate_factor', type=float, default=0.8, help='factor in which to decrease the lr at each lr_steps')
    parser.add_argument('--first_point', type=int, default=0, help='first data point to optimize')
    parser.add_argument('--last_point', type=int, default=None, help='last data point to optimize')
    parser.add_argument('--sampling', type=str, default='random', help='sampling strategy to use')
    parser.add_argument('--param_init_range', type=float, default=0.1, help='sampling range for parameter initialization')
    
    # Estimator configurations
    parser.add_argument('--server', type=str, default='local', help='which qiskit optimizer to use, local or runtime')
    parser.add_argument('--save_account', type=bool, default=False, help='save account credentials for first run')

    # Logger
    parser.add_argument('--logger', type=object, default=None, help='logger object')
    parser.add_argument('--logger_opt', type=object, default=None, help='logger optimizer object')
    
    config = parser.parse_args(args=[])

    return config