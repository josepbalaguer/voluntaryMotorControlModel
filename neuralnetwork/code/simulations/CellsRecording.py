from mpi4py import MPI
from neuron import h
from .Simulation import Simulation
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math



comm = MPI.COMM_WORLD
sizeComm = comm.Get_size()
rank = comm.Get_rank()

class CellsRecording(Simulation):
	""" Record cells membrane potential over time. """

	def __init__(self, parallelContext, cells, modelType, tStop = 100):
		""" Object initialization.

		Keyword arguments:
		parallelContext -- Neuron parallelContext object.
		cells -- dict containing lists of the objects we want to record (either all artificial cells or segments
			of real cells).
		modelType -- dictionary containing the model types ('real' or 'artificial') for every
			list of cells in cells.
		tStop -- Time in ms at which the simulation will stop (default = 100).
		"""

		Simulation.__init__(self, parallelContext)

		if rank==1:
			print("\nWarning: mpi execution in this simulation is not supported and therfore useless.")
			print("Only the results of the first process are considered...\n")

		self._cells = cells
		self._modelType = modelType
		self._set_tstop(tStop)
		# To plot the results with a high resolution we use an integration step equal to the Neuron dt
		self._set_integration_step(h.dt)

	"""
	Redefinition of inherited methods
	"""
	def _initialize(self):
		Simulation._initialize(self)
		# Initialize rec list
		self._initialize_states()

	def _update(self):
		""" Update simulation parameters. """
		for cellName in self._cells:
			if self._modelType[cellName] == "real":
				for i,cell in enumerate(self._cells[cellName]):
					self._states[cellName][i].append(cell(0.5).v)

					self._ina[cellName][i].append(cell(0.5).motoneuron.ina)
					self._ikrect[cellName][i].append(cell(0.5).motoneuron.ikrect)
					self._il[cellName][i].append(cell(0.5).motoneuron.il)
					self._ican[cellName][i].append(cell(0.5).motoneuron.icaN)
					self._ical[cellName][i].append(cell(0.5).motoneuron.icaL)
					self._ikca[cellName][i].append(cell(0.5).motoneuron.ikca)

			elif self._modelType[cellName] == "artificial":
				for i,cell in enumerate(self._cells[cellName]):
					self._states[cellName][i].append(cell.cell.M(0))

	def save_results(self):
		""" Save the simulation results. """
		print("Not implemented...use the plot method to visualize and save the plots")

	def plot(self,name="",title="",block=True):
		""" Plot the simulation results. """
		if rank == 0:

			fig = plt.figure(figsize=(16,7))
			gs = gridspec.GridSpec(self._nCells,1)
			ax = []
			fontsize=25
			plt.rcParams['font.sans-serif'] = "Arial"
			plt.rcParams['font.family'] = "sans-serif"
			

			cmap = plt.get_cmap('autumn')
			tStop = self._get_tstop()
			
			colors = cmap(np.linspace(0.1,0.9,self._nCells))
			for i,cellName in enumerate(self._states):
				for state in self._states[cellName]:
					ax.append(plt.subplot(gs[i]))
					ax[-1].plot(np.linspace(0,self._get_tstop(),len(state)),state,color=colors[i], linewidth=5)
					ax[-1].set_ylabel('Membrane potential (mV)', fontsize=fontsize)
			ax[-1].set_xlabel('Time (ms)', fontsize=fontsize)

			plt.gca().spines['top'].set_visible(False)
			plt.gca().spines['right'].set_visible(False)
			vmax=max(state)
			vmin=min(state)
			interval=abs(vmax-vmin)
			
			if interval<=50 and interval>=0.1:
				vmin=math.ceil(min(state)*10)/10
				vmax=math.floor(max(state)*10)/10
			elif interval<0.1:
				vmin=math.ceil(min(state)*100)/100
				vmax=math.floor(max(state)*100)/100
			elif interval>50:
				vmax=math.floor(max(state))
				vmin=math.ceil(min(state))
			
			interval=abs(vmax-vmin)
			plt.tick_params(direction='out')

			plt.xticks(np.arange(0,2*tStop,tStop))
			plt.yticks(np.arange(vmin,vmax+interval,interval))
			plt.yticks(fontsize= fontsize)
			plt.xticks(fontsize= fontsize)
			plt.margins(0)
			fileName = time.strftime("%Y_%m_%d_CellsRecording_"+name+".pdf")
			print(self._resultsFolder+fileName)
			plt.savefig(self._resultsFolder+fileName, format="pdf",transparent=True)
			#plt.show(block=block)

			cnt=1
			while cnt<=2:
				for i,cellName in enumerate(self._states):
					if cellName=="MnReal":
						totMn=(len(self._cells[cellName]))
						for state in self._states[cellName]:
							numbermn=cnt
							fileNametxt = time.strftime("%Y_%m_%d_CellsRecording_"+name+"_"+str(numbermn))
							np.savetxt(self._resultsFolder+fileNametxt, state)
							cnt=cnt+1

	def plot_ionchannels(self,name="",title="",block=True,ion_type=1):
		""" Plot the simulation results. """
		if rank == 0:

			fig = plt.figure(figsize=(16,7))
			gs = gridspec.GridSpec(self._nCells,1)
			ax = []
			fontsize=25
			plt.rcParams['font.sans-serif'] = "Arial"
			plt.rcParams['font.family'] = "sans-serif"


			cmap = plt.get_cmap('autumn')
			tStop = self._get_tstop()

			colors = cmap(np.linspace(0.1,0.9,self._nCells))

			if ion_type==1:
				self._ion=self._ina
				ion_name="Ina"
			elif ion_type==2:
				self._ion=self._ikrect
				ion_name = "Ikrect"
			elif ion_type == 3:
				self._ion=self._il
				ion_name = "Il"
			elif ion_type == 4:
				self._ion=self._ican
				ion_name = "Ican"
			elif ion_type == 5:
				self._ion = self._ical
				ion_name = "Ical"
			elif ion_type == 6:
				self._ion = self._ikca
				ion_name = "Ikca"

			for i,cellName in enumerate(self._ion):
				for ion in self._ion[cellName]:
					ax.append(plt.subplot(gs[i]))
					ax[-1].plot(np.linspace(0,self._get_tstop(),len(ion)),ion,color=colors[i], linewidth=5)
					ax[-1].set_ylabel('Ion current (mA)', fontsize=fontsize)
			ax[-1].set_xlabel('Time (ms)', fontsize=fontsize)

			plt.gca().spines['top'].set_visible(False)
			plt.gca().spines['right'].set_visible(False)
			vmax=max(ion)
			vmin=min(ion)
			interval=abs(vmax-vmin)
			plt.tick_params(direction='out')

			plt.xticks(np.arange(0,2*tStop,tStop))
			plt.yticks(np.arange(vmin,vmax+interval,interval))
			plt.yticks(fontsize= fontsize)
			plt.xticks(fontsize= fontsize)
			plt.margins(0)
			fileName = time.strftime("%Y_%m_%d_IonChannel_"+str(ion_name)+"_"+str(name)+".pdf")
			print(self._resultsFolder+fileName)
			plt.savefig(self._resultsFolder+fileName, format="pdf",transparent=True)

			numbermn=1
			for i, cellName in enumerate(self._ion):
				for ion in self._ion[cellName]:
					fileNametxt = time.strftime("%Y_%m_%d_IonChannel_"+str(ion_name)+"_"+str(name)+"_"+str(numbermn))
					np.savetxt(self._resultsFolder+fileNametxt, ion)
					numbermn=numbermn+1


	"""
	Specific Methods of this class
	"""

	def _initialize_states(self):
		self._states = {}

		self._ina = {}
		self._ikrect = {}
		self._il = {}
		self._ican = {}
		self._ical = {}
		self._ikca = {}

		self._nCells = len(list(self._cells.keys()))
		for cellName in self._cells:
			self._states[cellName] = []

			self._ina[cellName] = []
			self._ikrect[cellName] = []
			self._il[cellName] = []
			self._ican[cellName] = []
			self._ical[cellName] = []
			self._ikca[cellName] = []

			for cell in self._cells[cellName]:
				self._states[cellName].append([])

				self._ina[cellName].append([])
				self._ikrect[cellName].append([])
				self._il[cellName].append([])
				self._ican[cellName].append([])
				self._ical[cellName].append([])
				self._ikca[cellName].append([])
