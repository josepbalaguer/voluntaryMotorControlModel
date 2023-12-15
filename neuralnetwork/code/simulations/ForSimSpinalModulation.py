from mpi4py import MPI
from neuron import h
from .ForwardSimulation import ForwardSimulation
from .CellsRecording import CellsRecording
import time
import numpy as np
from tools import general_tools  as gt
import matplotlib.pyplot as plt
import pickle
from tools import seed_handler as sh
sh.set_seed()


comm = MPI.COMM_WORLD
sizeComm = comm.Get_size()
rank = comm.Get_rank()

class ForSimSpinalModulation(ForwardSimulation,CellsRecording):
	""" Integration of a NeuralNetwork object over time given an input.
		The simulation results are the cells membrane potential over time.
	"""

	def __init__(self, parallelContext, eesAmplitude, neuralNetwork, cells, modelType, frequency, afferentInput=None, eesObject=None, eesModulation=None, tStop = 100):
		""" Object initialization.

		Keyword arguments:
		parallelContext -- Neuron parallelContext object.
		neuralNetwork -- NeuralNetwork object.
		frequency -- firing rate of the supraspinal fibers.
		cells -- dict containing cells list (or node lists for real cells) from which we record the membrane potentials.
		modelType -- dictionary containing the model types ('real' or 'artificial') for every
			list of cells in cells.
		afferentInput -- Dictionary of lists for each type of fiber containing the
			fibers firing rate over time and the dt at wich the firing rate is updated.
			If no afferent input is desired use None (default = None).
		eesObject -- EES object connected to the NeuralNetwork, usefull for some plotting
			info and mandatory for eesModulation (Default = None).
		eesModulation -- possible dictionary with the following strucuture: {'modulation':
			dictionary containing a	signal of 0 and 1s used to activate/inactivate
			the stimulation for every muscle that we want to modulate (the dictionary
			keys have to be the muscle names used in the neural network structure), 'dt':
			modulation dt}. If no modulation of the EES is intended use None (default = None).
		tStop -- Time in ms at which the simulation will stop (default = 100). In case
			the time is set to -1 the neuralNetwork will be integrated for all the duration
			of the afferentInput.
		"""

		if rank==1:
			print("\nWarning: mpi execution in this simulation is not supported and therefore useless.")
			print("Only the results of the first process are considered...\n")
		CellsRecording.__init__(self, parallelContext, cells, modelType, tStop)

		ForwardSimulation.__init__(self,parallelContext, neuralNetwork, frequency, afferentInput, eesObject, eesModulation, tStop)
		self._set_integration_step(h.dt)


		self._nn=neuralNetwork
		self._eesAmplitude=eesAmplitude
		self._interval=0
		self._tStop=tStop
		self._frequency = frequency

	"""
	Redefinition of inherited methods
	"""

	def _initialize(self):
		ForwardSimulation._initialize(self)
		CellsRecording._initialize(self)

	def _update(self):
		""" Update simulation parameters. """

		CellsRecording._update(self)
		ForwardSimulation._update(self)

	def plot_membrane_potatial(self,name="",title="",block=False):
		CellsRecording.plot(self,name,title,block)

	def plot_ion_channel(self,name="",title="",block=False):
		CellsRecording.plot_ionchannels(self, name, title, block, 1)
		CellsRecording.plot_ionchannels(self, name, title, block, 2)
		CellsRecording.plot_ionchannels(self, name, title, block, 3)
		CellsRecording.plot_ionchannels(self, name, title, block, 4)
		CellsRecording.plot_ionchannels(self, name, title, block, 5)
		CellsRecording.plot_ionchannels(self, name, title, block, 6)

	def save_results(self,name):
		""" Save the simulation results. """
		if rank == 0:
			fileName = time.strftime("%Y_%m_%d_FSSM_nSpikes")+name+".p"
			with open(self._resultsFolder+fileName, 'w') as pickle_file:
				pickle.dump(self._nSpikes, pickle_file)
				pickle.dump(self._nActiveCells, pickle_file)
