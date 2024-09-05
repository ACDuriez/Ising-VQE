import os
import time
import json
from dataclasses import dataclass, field
from typing import List
import numpy as np
import networkx as nx
import pickle
import scipy

from qiskit.circuit import QuantumCircuit,ParameterVector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import L_BFGS_B,SPSA

from qiskit_ibm_runtime import Session, QiskitRuntimeService, OptionsV2,Session
from qiskit_ibm_runtime.fake_provider import FakeTorino,FakeManilaV2,FakeAlmadenV2
from qiskit_ibm_runtime import EstimatorV2

from qiskit.quantum_info import SparsePauliOp

from qiskit.primitives import Estimator,StatevectorEstimator

class Solver:
    def __init__(self, config: dict):
        # Physical system configuration
        self.num_sites = config.num_sites
        self.transition = config.transition
        self.hx = config.hx
        
        # Connectivity graph
        self._graph = self._get_line_graph(self.num_sites)
        
        #Quantum circuit
        self.layers = config.layers
        if config.ansatz == 'hva':
            self.ansatz = self.get_ansatz_hva()

        #Simulation configurations
        self.field_range = config.field_range
        self.field_steps = config.field_steps
        self._field_values = np.linspace(*self.field_range,self.field_steps)
        self.first_point = config.first_point
        self.last_point = config.last_point
        # Measured observables on groundstate
        self._aux_operators = [self._get_kk_op()] + self._get_local_mags_op() # observables:[kinks,mags1,...,magsL]

        # Optimization configurations
        self.opt_algorithm = config.opt_algorithm
        self.opt_iterations = config.opt_iterations
        self.learning_rate = config.learning_rate
        self.learning_rate_step = config.learning_rate_step
        self.learning_rate_factor = config.learning_rate_factor
        self.perturbation_magnitude = config.perturbation_magnitude
        self.sampling = config.sampling
        self.param_init_range = config.param_init_range

        # Runtime configurations
        self.backend_name = config.backend_name
        self.save_account = config.save_account

        # Estimator configurations
        self.service = None
        self.backend = None
        self.session = None
        self.server = config.server
        # self._estimator = self._set_estimator()

        # Data points
        if not config.data_points:
            self.data_points = [dict(idx=idx,
                                     field_value=field_value,
                                     exact_energy=0.,
                                     exact_observables=np.zeros(len(self._aux_operators)),
                                     vqe_energy=0.,
                                     vqe_observables=np.zeros(len(self._aux_operators)),
                                     optimal_parameters=np.zeros(self.ansatz.num_parameters),
                                     optimization_data=[],
                                     optimization_config=[],
                                     ) for idx,field_value in enumerate(self._field_values)]
            # self.data_points = [SimulationResult(idx=idx,field_value=field_value) for idx,field_value in enumerate(self._field_values)]
        else:
            self.data_points = config.data_points

        # Logger
        self.logger = config.logger
        self.logger_opt = config.logger_opt

        # Saving directories
        self.data_dir_path = config.data_dir_path
        self.img_dir_path = config.img_dir_path
        
    def save_class(self):
        pickle.dump(self,open('solver.pkl','wb'))

    def _set_runtime(self):
        if self.save_account == True:
            cloud_credentials = json.load(open('cloud_crentials.json','r'))
            QiskitRuntimeService.save_account(channel="ibm_cloud", 
                                              overwrite=True, 
                                              token=cloud_credentials['token'], 
                                              instance=cloud_credentials['instance'], 
                                              set_as_default=True,
                                              name=cloud_credentials['name'])
        
        service = QiskitRuntimeService(name="alan-runtime")
        if self.backend_name=='fake_backend':
            backend = FakeManilaV2()
        return service, backend

    @staticmethod
    def _get_line_graph(num_sites):
        graph_line = nx.Graph()
        graph_line.add_nodes_from(range(num_sites))

        edge_list = []
        for i in graph_line.nodes:
            if i < num_sites-1:
                edge_list.append((i,i+1))

        # Generate graph from the list of edges
        graph_line.add_edges_from(edge_list)
        return graph_line

    def _get_hamiltonian(self,field):
        """Returns the hamiltonian in the case where hl=hr. Here the varying field is 
        the antiparallel field at the borders."""
        sparse_list = []
        # Uniform X field
        for qubit in self._graph.nodes():
            coeff = ('X',[qubit],-1*self.hx)
            sparse_list.append(coeff)

        # Anti-paralel field at the borders
        coeff = ('Z',[0],field) #this is the positive field (order reversed)
        sparse_list.append(coeff)
        coeff = ('Z',[self.num_sites-1],-1.*field)
        sparse_list.append(coeff)

        #Interaction field (ZZ)
        for i,j in self._graph.edges():
            coeff = ('ZZ',[i,j],-1.)
            sparse_list.append(coeff)
        
        hamiltonian = SparsePauliOp.from_sparse_list(sparse_list,num_qubits=self.num_sites)
        return hamiltonian

    def _get_kk_op(self):
        sparse_list = []
        for i,j in self._graph.edges():
            coeff = ('II',[i,j],0.5)
            sparse_list.append(coeff)       
            coeff = ('ZZ',[i,j],-0.5)
            sparse_list.append(coeff)
        
        kk_op = SparsePauliOp.from_sparse_list(sparse_list,num_qubits=self.num_sites)
        return kk_op

    def _get_local_mags_op(self):
        """Returns the local magnetization operator of each
        magnetization for each spin.
        """
        mag_op_list = [SparsePauliOp.from_sparse_list([('Z',[qubit],1.0)],num_qubits=self.num_sites) for qubit in self._graph.nodes()]

        return mag_op_list

    def get_ansatz_hva(self):
        """Creates the hamiltonian variaitonal ansatz for a given
        lattice graph and number of layers.
        Args:
            graph: lattice graph
            theta_list: list of parameters
        """
        theta_list = ParameterVector('θ',3*self.layers)

        circuit = QuantumCircuit(self.num_sites)
        
        even_edges = [edge for edge in self._graph.edges() if edge[0]%2==0]
        odd_edges = [edge for edge in self._graph.edges() if edge[0]%2!=0]
        
        # initial_state
        circuit.h(range(self.num_sites))
        for layer_index in range(self.layers):

            # Coupling term
            for pair in even_edges:
                circuit.rzz(2 * theta_list[3*layer_index],pair[0],pair[1])
            for pair in odd_edges:
                circuit.rzz(2 * theta_list[3*layer_index],pair[0],pair[1])
            # boundary field term
            circuit.rz(2 *theta_list[3*layer_index+2],0)
            circuit.rz(-2 * theta_list[3*layer_index+2], self.num_sites-1) 
            # transverse field term
            circuit.rx(2 * theta_list[3*layer_index+1], range(self.num_sites))

        return circuit

    def _compute_observables(self,hamiltonian,calculator,aux_operators=None,complete_result=False):

        # import ipdb; ipdb.set_trace()
        results = calculator.compute_minimum_eigenvalue(hamiltonian,aux_operators=aux_operators)
        if complete_result:
            return results
        else:
            energy_value = results.eigenvalue
            if aux_operators is None:
                return energy_value
            else:
                aux_operators_evaluated = np.real([operator_result[0] for operator_result in results.aux_operators_evaluated])
                return energy_value,aux_operators_evaluated
            
    def _compute_exact_observables(self,hamiltonian,calculator,aux_operators=None,complete_result=False):

        # import ipdb; ipdb.set_trace()
        results = calculator.compute_minimum_eigenvalue(hamiltonian,aux_operators=aux_operators)
        if complete_result:
            return results
        else:
            energy_value = results.eigenvalue
            if aux_operators is None:
                return energy_value
            else:
                aux_operators_evaluated = np.real([operator_result[0] for operator_result in results.aux_operators_evaluated])
                return energy_value,aux_operators_evaluated

    def _compute_exact_result(self):
        calculator = NumPyMinimumEigensolver()
        for data_point in self.data_points[self.first_point:self.last_point]:
            hamiltonian = self._get_hamiltonian(data_point['field_value'])
            data_point['exact_energy'],data_point['exact_observables'] = self._compute_exact_observables(hamiltonian,
                                                                                                 calculator,
                                                                                                 self._aux_operators)

    def _optimize_points(self,estimator,ansatz_isa):
        """
        Optimizes the VQE circuit from the first to the last informed
        datapoint.
        Args:
        sampling: 
            'previous' -> taking optimized previous point for initialization 
            'random'   -> samples uniformly for the range
        """
        
        def cost_func(params):
            """Return estimate of energy from estimator

            Parameters:
                params (ndarray): Array of ansatz parameters

            Returns:
                float: Energy estimate
            """
            pub = (ansatz_isa, [hamiltonian_isa], [params])
            result = estimator.run(pubs=[pub]).result()
            energy = result[0].data.evs[0]

            cost_history_dict["iters"] += 1
            cost_history_dict["prev_vector"] = params
            cost_history_dict["cost_history"].append(energy)
            self.logger.info(f"Iters. done: {cost_history_dict['iters']} [Current cost: {energy}]")

            return energy
        
        # optimizer = L_BFGS_B(maxiter=1,maxfun=5)
        # import ipdb;ipdb.set_trace()
        
        learning_rate,perturbation = None,None
        for data_point in self.data_points[self.first_point:self.last_point]:
            self.logger.info(f'Optimizing data point {data_point['idx']}')
            cost_history_dict = {"prev_vector": None,
                                 "iters": 0,
                                 "cost_history": []}
            
            if data_point['idx'] == 0 or self.sampling=='random' or self.sampling is None:
                initial_point = np.random.uniform(-self.param_init_range,self.param_init_range,size=self.ansatz.num_parameters)
                sampling = 'random'
            else:
                initial_point = self.data_points[data_point['idx']-1]['optimal_parameters']
                sampling='previous'
            
            # Defining hamiltonian for point
            hamiltonian_isa = self._get_hamiltonian(data_point['field_value']).apply_layout(layout=ansatz_isa.layout)

            optimizer = SPSA(self.opt_iterations)
            if data_point['idx'] == 0:
                self.logger.info(f'Calibrating optimizer for {sampling} point...')
                init_learning_rate,init_perturbation = optimizer.calibrate(cost_func,initial_point,c=self.perturbation_magnitude)
                optimizer.learning_rate = init_learning_rate
                optimizer.perturbation = init_perturbation
            elif learning_rate==None or perturbation==None:
                self.logger.info(f'Calibrating optimizer for {sampling} point...')
                learning_rate,perturbation = optimizer.calibrate(cost_func,initial_point,c=1e-3,target_magnitude=1e-2)
                optimizer.learning_rate = learning_rate
                optimizer.perturbation = perturbation
            else:
                optimizer.learning_rate = learning_rate
                optimizer.perturbation = perturbation
            
            self.logger.info(f'Calibration done! Optimizing point {data_point['idx']}')
            # Main calculation
            start_time = time.time()
            result = optimizer.minimize(fun=cost_func,
                                        x0=initial_point,)
            # result = scipy.optimize.minimize(fun=cost_func,
            #                             x0=initial_point,method='COBYLA',options={'maxiter':2})
            end_time = time.time()
            self.logger.info(f'\t Optimization time: {end_time - start_time}')

            # Storing result
            data_point['vqe_energy'] = result.fun
            data_point['optimal_parameters'] = result.x
            # Optimization history
            for k in cost_history_dict.keys():
                if k!='iters':
                    cost_history_dict[k] = np.array(cost_history_dict[k].copy())


            data_point['optimization_data'].append(cost_history_dict)
            data_point['optimization_config'].append(dict(opt_algortihm=self.opt_algorithm,
                                                     iterations=self.opt_iterations,
                                                     learning_rate=self.learning_rate,
                                                     learning_rate_step=self.learning_rate_step,
                                                     learning_rate_factor = self.learning_rate_factor,
                                                     sampling=sampling))
            
            self.save_class()

    def compute_observables_vqe(self,estimator,ansatz_isa):
        data_points = self.data_points[self.first_point:self.last_point]
        for i,observable in enumerate(self._aux_operators):
            # circuits = [self.ansatz.assign_parameters(data_point['optimal_parameters']) for data_point in data_points]
            # observables = [observable]*len(circuits)
            # # import ipdb; ipdb.set_trace()
            # result = estimator.run(circuits,observables).result().values
            # # import ipdb; ipdb.set_trace()
            # for j,data_point in enumerate(data_points):
            #     data_point['vqe_observables'][i] = result[j]
            
            # import ipdb;ipdb.set_trace()
            observables = [[aux_operator] for aux_operator in self._aux_operators]
            observables = [
                [observable.apply_layout(ansatz_isa.layout) for observable in observable_set]
                for observable_set in observables
            ]
            params = np.vstack([data_point['optimal_parameters'] for data_point in data_points]).T
            pub = (ansatz_isa,observables,params)
            result = estimator.run([pub])
            data = result[0].data

    def run(self):
        """
        Runs the whole simulation
        """

        self.logger.info('Beggining simulation; Computig exact results...')
        
        self._compute_exact_result()
        self.logger.info('Exact results calculated! Now optimizing points...')

        # Setting runtime credentials
        _,backend = self._set_runtime()    
        
        # Transpilation options
        pm = generate_preset_pass_manager(target=backend.target, optimization_level=3)
        ansatz_isa = pm.run(self.ansatz)

        # Setting optimizer
        with Session(backend=backend) as session:
            if self.server=='local':
                estimator = StatevectorEstimator()
                # estimator = EstimatorV2(session=session)
                # estimator.options.default_shots = 10000
                # estimator.options.resilience_level = 2
            
            self._optimize_points(estimator,ansatz_isa=ansatz_isa)
            
            self.logger.info('VQE optimization completed! Now computing vqe observables')
            self.save_class()
        
            # self.compute_observables_vqe(estimator,ansatz_isa=ansatz_isa)
            session.close()
        
        self.logger.info('VQE observables computed, session closed! Saving data.')
        
        self.save_class()

        self.logger.info('Done!')     