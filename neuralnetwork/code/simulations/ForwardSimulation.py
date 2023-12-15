from mpi4py import MPI
from neuron import h
from .Simulation import Simulation
from cells import AfferentFiber
import time
import numpy as np
from tools import firings_tools as tlsf
import pickle
from tools import seed_handler as sh
import scipy.io
sh.set_seed()

comm = MPI.COMM_WORLD
sizeComm = comm.Get_size()
rank = comm.Get_rank()

class ForwardSimulation(Simulation):
	""" Integration of a NeuralNetwork object over time given an input (ees or afferent input or both). """

	def __init__(self, parallelContext, neuralNetwork, frequency, afferentInput=None, eesObject=None, eesModulation=None, tStop = 100):
		""" Object initialization.

		Keyword arguments:
		parallelContext -- Neuron parallelContext object.
		neuralNetwork -- NeuralNetwork object.
		frequency -- firing rate of the supraspinal fibers.
		afferentInput -- Dictionary of lists for each type of fiber containing the
			fibers firing rate over time and the dt at wich the firing rate is updated.
			If no afferent input is desired, use None (default = None).
		eesObject -- EES object connected to the NeuralNetwork,
			mandatory for eesModulation (Default = None).
		eesModulation -- possible dictionary with the following strucuture: {'modulation':
			dictionary containing a	signal of 0 and 1s used to activate/inactivate
			the stimulation for every muscle that we want to modulate (the dictionary
			keys have to be the muscle names used in the neural network structure), 'dt':
			modulation dt}. If no modulation of the EES is intended use None (default = None).
		tStop -- Time in ms at wich the simulation will stop (default = 100). In case
			the time is set to -1 the neuralNetwork will be integrated for all the duration
			of the afferentInput.
		"""

		Simulation.__init__(self,parallelContext)

		if rank==1:
			print("\nMPI execution: the cells are divided in the different hosts\n")
		self._frequency = frequency
		self._nn = neuralNetwork
		if 'Iaf' in self._nn.get_primary_afferents_names():
			self._Iaf = self._nn.get_primary_afferents_names()[0] if self._nn.get_primary_afferents_names() else []
		self._IIf = self._nn.get_secondary_afferents_names()[0] if self._nn.get_secondary_afferents_names() else []
		self._Mn = self._nn.get_motoneurons_names() if self._nn.get_motoneurons_names() else []
		self._IntMn = self._nn.get_intf_motoneurons_names() if self._nn.get_intf_motoneurons_names() else []
		self._CM = self._nn.get_corticomotoneurons_names()[0] if self._nn.get_corticomotoneurons_names() else []
		self._fineTask = self._nn._fineTask if self._nn.get_corticomotoneurons_names()[0] else []

		self._set_integration_step(AfferentFiber.get_update_period())

		# Initialization of the afferent modulation
		if afferentInput == None:
			self._afferentModulation = False
			self._afferentInput = None
			if tStop>0: self._set_tstop(tStop)
			else : raise Exception("If no afferents input are provided tStop has to be greater than 0.")
		else:
			self._afferentModulation = True
			self._afferentInput = afferentInput[0]
			self._dtUpdateAfferent = afferentInput[1]
			self._init_afferents_fr()

			key = []
			key.append(list(self._afferentInput.keys())) #e.g. dict_keys(['BIC', 'TRI'])
			key.append(list(self._afferentInput[key[0][1]].keys()))  # e.g. dict_keys(['Iaf', 'IIf']). key[0][1] is accessing the muscles and obtaining the keys in the second muscle 'TRI'
			if 'Iaf' in self._afferentInput[key[0][1]].keys():
				self._inputDuration = len(self._afferentInput[key[0][1]][key[1][0]]) * self._dtUpdateAfferent  # Length from muscle 1 and primary fibers
				if tStop == -1 or tStop>= self._inputDuration: self._set_tstop(self._inputDuration-self._dtUpdateAfferent)
				else: self._set_tstop(tStop)
			else:
				self._set_tstop(tStop)

		self._ees = eesObject
		# Initialization of the binary stim modulation
		if eesModulation == None or eesObject == None:
			self._eesBinaryModulation = False
			self._eesProportionalModulation = False
			self._eesParam = {'state':None, 'amp':None, 'modulation':None, 'dt':None}
		elif eesModulation['type']=="binary":
			self._eesBinaryModulation = True
			self._eesProportionalModulation = False
			current, percIf, percIIf, percMn = self._ees.get_amplitude()
			self._eesParam = {'state':{}, 'modulation':eesModulation['modulation'], 'dt':eesModulation['dt']}
			self._eesParam['amp'] = [percIf, percIIf, percMn]
			for muscle in eesModulation['modulation']:
				self._eesParam['state'][muscle] = 1
		elif eesModulation['type']=="proportional":
			self._eesBinaryModulation = False
			self._eesProportionalModulation = True
			self._eesParam = {'modulation':eesModulation['modulation'], 'dt':eesModulation['dt']}
			current, percIf, percIIf, percMn = self._ees.get_amplitude()
			self._eesParam['maxAmp'] = np.array([percIf, percIIf, percMn])

		#Initialization of the result dictionaries
		self._meanFr = None
		self._estimatedEMG = None
		self._nSpikes = None
		self._nActiveCells = None

	"""
	Redefinition of inherited methods
	"""
	def _initialize(self):
		Simulation._initialize(self)
		self._init_aff_fibers(self._get_tstop(),self._get_integration_step())
		self._timeUpdateAfferentsFr = 0
		self._timeUpdateEES = 0

	def _update(self):
		""" Update simulation parameters. """
		comm.Barrier()
		self._nn.update_afferents_ap(h.t)
		if self._afferentModulation:
			if h.t-self._timeUpdateAfferentsFr>= (self._dtUpdateAfferent-0.5*self._get_integration_step()):
				self._timeUpdateAfferentsFr = h.t
				self._set_afferents_fr(int(h.t/self._dtUpdateAfferent))
				self._nn.set_afferents_fr(self._afferentFr)

		if self._eesBinaryModulation:
			if h.t-self._timeUpdateEES>= (self._eesParam['dt']-0.5*self._get_integration_step()):
				ind = int(h.t/self._eesParam['dt'])
				for muscle in self._eesParam['modulation']:
					if self._eesParam['state'][muscle] != self._eesParam['modulation'][muscle][ind]:
						if self._eesParam['state'][muscle] == 0: self._ees.set_amplitude(self._eesParam['amp'],[muscle])
						else: self._ees.set_amplitude([0,0,0],[muscle])
						self._eesParam['state'][muscle] = self._eesParam['modulation'][muscle][ind]

		if self._eesProportionalModulation:
			if h.t-self._timeUpdateEES>= (self._eesParam['dt']-0.5*self._get_integration_step()):
				ind = int(h.t/self._eesParam['dt'])
				for muscle in self._eesParam['modulation']:
					amp = list(self._eesParam['maxAmp']*self._eesParam['modulation'][muscle][ind])
					self._ees.set_amplitude(amp,[muscle])

	def _end_integration(self):
		""" Print the total simulation time and extract the results. """
		Simulation._end_integration(self)
		self._extract_results()

	"""
	Specific Methods of this class
	"""
	def _init_aff_fibers(self,tStop,step):
		""" Return the percentage of afferent action potentials erased by the stimulation. """
		for muscleName in self._nn.cells:
			for cellName in self._nn.cells[muscleName]:
				if cellName in self._nn.get_afferents_names():
					for fiber in self._nn.cells[muscleName][cellName]:
						fiber.initialise()
				if self._fineTask:
					if cellName in self._nn.get_corticomotoneurons_names():
						for cell in self._nn.cells[muscleName][cellName]:
							cell.create_poisson_firing(self._frequency,tStop,step)


	def _init_afferents_fr(self):
		""" Initialize the dictionary necessary to update the afferent fibers. """
		self._afferentFr = {}
		for muscle in self._afferentInput:
			self._afferentFr[muscle]={}
			for cellType in self._afferentInput[muscle]:
				if cellType in self._nn.get_afferents_names():
					self._afferentFr[muscle][cellType]= 0.
				else: raise Exception("Wrong afferent input structure!")

	def _set_afferents_fr(self,i):
		""" Set the desired firing rate in the _afferentFr dictionary. """
		for muscle in self._afferentInput:
			for cellType in self._afferentInput[muscle]:
				self._afferentFr[muscle][cellType] = self._afferentInput[muscle][cellType][i]

	def _extract_results(self,samplingRate = 1000.):
		""" Extract the simulation results. """
		if rank==0: print("Extracting the results... ", end=' ')
		self._firings = {}
		self._meanFr = {}
		self._estimatedEMG = {}
		self._nSpikes = {}
		self._nActiveCells = {}
		for muscle in self._nn.actionPotentials:
			self._firings[muscle]={}
			self._meanFr[muscle]={}
			self._estimatedEMG[muscle]={}
			self._nSpikes[muscle]={}
			self._nActiveCells[muscle]={}
			for cell in self._nn.actionPotentials[muscle]:
				self._firings[muscle][cell] = tlsf.exctract_firings(self._nn.actionPotentials[muscle][cell],self._get_tstop(),samplingRate)
				if rank==0: self._nActiveCells[muscle][cell] = np.count_nonzero(np.sum(self._firings[muscle][cell],axis=1))
				self._nSpikes[muscle][cell] = np.sum(self._firings[muscle][cell])
				self._meanFr[muscle][cell] = tlsf.compute_mean_firing_rate(self._firings[muscle][cell],samplingRate)
				if cell in self._nn.get_motoneurons_names():
					self._estimatedEMG[muscle][cell] = tlsf.synth_rat_emg(self._firings[muscle][cell],samplingRate)
		if rank==0: print("...completed.")

	def get_estimated_emg(self,muscleName):
		emg = [self._estimatedEMG[muscleName][mnName] for mnName in self._Mn]
		emg = np.sum(emg,axis=0)
		return emg

	def get_estimated_emg_IntMn(self,muscleName):
		emg = [self._estimatedEMG[muscleName][mnName] for mnName in self._IntMn]
		emg = np.sum(emg,axis=0)
		return emg

	def get_mn_spikes_profile(self,muscleName):
		spikesProfile = [self._firings[muscleName][mnName] for mnName in self._Mn]
		spikesProfileall = np.sum(spikesProfile,axis=0)
		spikesProfileall = np.sum(spikesProfileall,axis=0)
		return spikesProfile,spikesProfileall

	def get_intmn_spikes_profile(self,muscleName):
		spikesProfile = [self._firings[muscleName][mnName] for mnName in self._IntMn]
		spikesProfileall = np.sum(spikesProfile,axis=0)
		spikesProfileall = np.sum(spikesProfileall,axis=0)
		return spikesProfile,spikesProfileall

	def get_iaf_spikes_profile(self,muscleName):
		spikesProfile = [self._firings[muscleName][self._Iaf]]
		spikesProfileall = np.sum(spikesProfile,axis=0)
		spikesProfileall = np.sum(spikesProfileall,axis=0)
		return spikesProfile,spikesProfileall

	def get_cm_spikes_profile(self,muscleName):
		spikesProfile = [self._firings[muscleName][self._CM]]
		spikesProfileall = np.sum(spikesProfile, axis=0)
		spikesProfileall = np.sum(spikesProfileall, axis=0)
		return spikesProfile, spikesProfileall

	def get_cm_spikes_poisson(self,muscleName):
		self._cm_poisson=[]
		for cell in self._nn.cells[muscleName][self._CM]:
			self._cm_poisson.append(cell._poisson_firings)
		return self._cm_poisson

	def get_weights_Iaf(self):
		self.weight_Iaf=self._nn.get_weight_Iaf_info()
		return self.weight_Iaf

	def get_weights_CST(self,muscleName):
		weight_CST_all=self._nn.get_weight_CST_info()
		ncst=int(len(self._nn.cells[muscleName][self._CM]))
		self.weight_CST=np.reshape(np.array(weight_CST_all),(int(len(weight_CST_all)/ncst),ncst))
		return self.weight_CST

	def get_dendrites_Iaf(self,muscleName):
		dendritenumber_Iaf_all=[]
		dendritesegment_Iaf_all=[]
		for cellName in self._nn.cells[muscleName]:
			if cellName in self._Mn:
				for cell in self._nn.cells[muscleName][cellName]:
					[dendritenumber_Iaf,dendritesegment_Iaf]=cell.get_dendrite_Iaf()
					dendritenumber_Iaf_all.append(dendritenumber_Iaf)
					dendritesegment_Iaf_all.append(dendritesegment_Iaf)
		return [dendritenumber_Iaf_all,dendritesegment_Iaf_all]

	def _get_perc_aff_ap_erased(self,muscleName,cellName):
		""" Return the percentage of afferent action potentials erased by the stimulation. """
		if cellName in self._nn.get_afferents_names():
			percErasedAp = []
			meanPercErasedAp = None
			for fiber in self._nn.cells[muscleName][cellName]:
				sent,arrived,collisions,perc = fiber.get_stats()
				percErasedAp.append(perc)
			percErasedAp = comm.gather(percErasedAp,root=0)

			if rank==0:
				percErasedAp = sum(percErasedAp,[])
				meanPercErasedAp = np.array(percErasedAp).mean()

			meanPercErasedAp = comm.bcast(meanPercErasedAp,root=0)
			percErasedAp = comm.bcast(percErasedAp,root=0)
			return meanPercErasedAp,percErasedAp
		else: raise Exception("The selected cell is not and afferent fiber!")

	def get_results(self,nn,muscleName,ees_pulses,name, eesAmplitude, Frequency,ICfreq,ICstimAmplitude,simTime,program,infocode,ic_pulses,mnreal):
		mEmg = []
		mSpikes = []
		mSpikes_all = []
		mSpikes_Iaf = []
		mSpikes_Iaf_all=[]
		mSpikes_CM = []
		mSpikes_CM_all = []
		mSpikes_CM_lambda_all = []
		nSamplesToAnalyse = -100  # last 100 samples
		if mnreal:
			# Extract emg responses
			try:
				mEmg.append(self.get_estimated_emg(muscleName[nSamplesToAnalyse:]))
			except (ValueError, TypeError) as error:
				mEmg.append(np.zeros(abs(nSamplesToAnalyse)))
			# Extract mn spikes
			try:
				fr_all, fr_pop = self.get_mn_spikes_profile(muscleName)
				mSpikes.append(fr_pop)
				mSpikes_all.append(fr_all)
			except (ValueError, TypeError) as error:
				mSpikes.append(np.zeros(abs(nSamplesToAnalyse)))
		else:
			# Extract emg responses
			try:
				mEmg.append(self.get_estimated_emg_IntMn(muscleName[nSamplesToAnalyse:]))
			except (ValueError, TypeError) as error:
				mEmg.append(np.zeros(abs(nSamplesToAnalyse)))
			# Extract mn spikes
			try:
				fr_all, fr_pop = self.get_intmn_spikes_profile(muscleName)
				mSpikes.append(fr_pop)
				mSpikes_all.append(fr_all)
			except (ValueError, TypeError) as error:
				mSpikes.append(np.zeros(abs(nSamplesToAnalyse)))

		# Extract Iaf spikes
		if 'Iaf' in nn.get_primary_afferents_names():
			[dendritenumber_Iaf, dendritesegment_Iaf] = self.get_dendrites_Iaf(muscleName)
			try:
				fr_Iaf_all, fr_Iaf_pop = self.get_iaf_spikes_profile(muscleName)
				mSpikes_Iaf.append(fr_Iaf_pop)
				mSpikes_Iaf_all.append(fr_Iaf_all)

				#Extract weights
				weight_Iaf=self.get_weights_Iaf()
				[dendritenumber_Iaf,dendritesegment_Iaf]=self.get_dendrites_Iaf(muscleName)
			except (ValueError, TypeError) as error:
				mSpikes_Iaf.append(np.zeros(abs(nSamplesToAnalyse)))
		else:
			mSpikes_Iaf=[0,0]
			mSpikes_Iaf_all=[0.0]
			weight_Iaf=0
			dendritenumber_Iaf=0
			dendritesegment_Iaf=0


		# Extract CM spikes
		if 'CM' in nn.get_corticomotoneurons_names():
			try:
				fr_CM_all, fr_CM_pop = self.get_cm_spikes_profile(muscleName)
				mSpikes_CM.append(fr_CM_pop)
				mSpikes_CM_all.append(fr_CM_all)
				weight_CST = self.get_weights_CST(muscleName)

				#Extract weights
			except (ValueError, TypeError) as error:
				mSpikes_CM.append(np.zeros(abs(nSamplesToAnalyse)))

		else:
			mSpikes_CM=[0,0]
			mSpikes_CM_all=[0,0]
			weight_CST=0

		# plot mn membrane potentials
		try:
			fileName = "%s_amp_%d_freq_%.2f" % (name, eesAmplitude, Frequency)
		except:
			fileName = "%s_amp_%.2f_freq_%.2f" % (name, eesAmplitude[0], Frequency)

		resultsFolder = "../../results/"
		fileNamemat = time.strftime("%Y_%m_%d_" + fileName + ".mat")
		scipy.io.savemat(resultsFolder + fileNamemat,
						 dict(raster_Mn=mSpikes[-1], raster_Mn_all=mSpikes_all[-1],
						 raster_Iaf=mSpikes_Iaf[-1],raster_Iaf_all=mSpikes_Iaf_all[-1],
							  raster_CM=mSpikes_CM[-1],raster_CM_all=mSpikes_CM_all[-1],
							  callprogram=program,infosimulation=infocode,weightIaf=weight_Iaf,weightCST_info=weight_CST,dendritenum_Iaf=dendritenumber_Iaf,dendriteseg_Iaf=dendritesegment_Iaf, raster_SCS=ees_pulses))

		resultsFolder_p = "../../results_p/"
		fileNamep = time.strftime("%Y_%m_%d_" + fileName + ".p")
		data={}
		data['scsampllitude']=eesAmplitude
		data['scsfrequency'] = Frequency
		data['ees_pulses']=ees_pulses
		data['icamplitude']=ICstimAmplitude
		data['icfrequency']=ICfreq
		data['icpulses']=ic_pulses
		data['lambda_IC_all'] = mSpikes_CM_lambda_all
		data["mEmg"]=mEmg
		data['mSpikes']=mSpikes
		data['mSpikes_all'] = mSpikes_all
		data['mSpikes_Iaf']=mSpikes_Iaf
		data['mSpikes_Iaf_all']=mSpikes_Iaf_all
		data['mSpikes_CM']=mSpikes_CM
		data['mSpikes_CM_all'] =mSpikes_CM_all
		data['drug5ht']=nn._drug
		data['simtime']=simTime
		data['name']=name
		data['callprogram']=program
		data['infosimulation']=infocode
		f=open(resultsFolder_p+fileNamep,'wb')
		pickle.dump(data, f)
		f.close()
