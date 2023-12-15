from .Cell import Cell
from neuron import h
import random as rnd
import numpy as np
import math
from mpi4py import MPI
from tools import seed_handler as sh
from tick.base import TimeFunction
from tick.plot import plot_point_process
from tick.hawkes import SimuInhomogeneousPoisson
sh.set_seed()

comm = MPI.COMM_WORLD
sizeComm = comm.Get_size()
rank = comm.Get_rank()

class CSFiber(Cell):
	""" Model of the supraspinal fiber (adapted from the afferent fiber model).

	The model integrates the collision of natural spikes with the ones
	induced by epidural electrical stimulation (EES) of the spinal cord.
	In particular the APs induced by the stimulation and the ones induced
	by the sensory organ are added to relative lists containing all APs
	positions in the fiber at the currant time. Every 0.1 ms (__updatePeriod)
	the position of the APs in the fiber has to be updated in order to
	simulate the propagation of the APs and the collision of natural and EES
	induced spikes. A refractory period of mean 1.6 ms and std of 0.16 ms is modeled.
	Note that the __updatePeriod can be increased in order to speed up the
	simulations. However, by increasing this value we would also lose resolution
	of the refractory period.
	"""

	__updatePeriod = 0.1 # Time period in ms between calls of the update fcn
	__eesWeight = -3 # Weight of a connection between an ees object and this cell
	__maxEesFrequency = 1001

	def __init__(self,delay,fineTask):
		""" Object initialization.

		Keyword arguments:
		delay -- time delay in ms needed by a spike to travel the whole fiber
		"""

		Cell.__init__(self)
		self._debug = False

		#Initialise cell parameters
		self._set_delay(delay)

		self.maxFiringRate = 200 # This should be lower than the frequency allowed by the refractory period
		self._maxSensorySpikesXtime = int(float(self._delay)/1000.*float(self.maxFiringRate)+2)
		self._maxEesSpikesXtime = int(float(self._delay)/1000.*float(self.__class__.__maxEesFrequency)+2)
		#Mean refractory period of 1.6 ms - 625 Hz
		noisePerc = 0.1
		self._refractoryPeriod = rnd.normalvariate(1.6,1.6*noisePerc)
		if self._refractoryPeriod>1000./self.maxFiringRate:
			self._refractoryPeriod=1000./self.maxFiringRate
			print("Warning: refractory period bigger than period between 2 natural pulses")
		#Position along the fiber recruited by the stimulation
		self._stimPosition = self._delay-0.5

		self.initialise()
		#Create an ARTIFICIAL_CELL Neuron mechanism that will be the source of a netCon object.
		#This will be used to comunicate the APs to target cells
		self.cell = h.CSFiber()

		#Create a netcon to make the fiber fire
		self._fire = h.NetCon(None,self.cell)

		# Boolean flag to record a segment of the fiber over time
		self._record = False
		self._fineTask = fineTask


	"""
	Specific Methods of this class
	"""
	def initialise(self,lastSpikeTime=0):
		""" Initialise the fiber. """

		#Initial firing rate of .1 Hz
		self._interval = 9999.
		self._oldFr = 0.
		self._lastNaturalSpikeTime = lastSpikeTime
		self._oldTime = 0.
		# tolerance to check for events
		self._tolerance = self.__class__.__updatePeriod/10.
		#Create list containing the natural spikes
		self._naturalSpikes = [None]*self._maxSensorySpikesXtime
		#Create list containing the EES induced spikes
		self._eesSpikes = [None]*self._maxEesSpikesXtime
		#Last spike in stim position
		self._lastStimPosSpikeTime = -9999.
		#Stats
		self._nCollisions = 0
		self._nNaturalSent = 0
		self._nNaturalArrived = 0
		
		self._rasterlastSpikes = []
		self._rasterfire=0
		self._firetime=0
		self._lastNaturalSpikeTime_matrix=[]

		self._index = 0

	# The delay correspond to the value naturalSpikes[] should have before being sent
	def _set_delay(self,delay):
		""" Set the delay.

		Keyword arguments:
		delay -- time delay in ms needed by a spike to travel the whole fiber
		"""

		minDelay = 1
		maxDelay = 100
		if delay>=minDelay and delay<=maxDelay:
			self._delay=delay
		else:
			raise Exception("Afferent fiber delay out of limits")

	def set_firing_rate(self, fr, noise=True):
		""" Set the afferent firing rate.

		Keyword arguments:
		fr -- firing rate in Hz
		"""

		if fr == self._oldFr: return
		if fr<=0:
			self._interval = 99999.
		elif fr>=self.maxFiringRate:
			self._interval = 1000.0/self.maxFiringRate
		elif fr<self.maxFiringRate and noise:
			mean = 1000.0/fr #ms
			sigma = mean*0.2
			self._interval = rnd.normalvariate(mean,sigma)
		else: self._interval = 1000.0/fr #ms
		self._oldFr = fr

		# Check whether after setting the new fr the fiber is ready to fire
		if (h.t-self._lastNaturalSpikeTime)>=self._interval-(self.__class__.__updatePeriod/2.):
			# In this case, shift a bit randomly the last natural spike time in order to reduce an
			# artificially induced synchronized activity between the different modeled fibers
			self._lastNaturalSpikeTime = h.t-np.random.uniform(self._interval/2.,self._interval,1)

	def update(self,time):
		self._update_activity(time)
		if self._record: self._record_segment(time)

	def _update_activity(self,time):
		""" Update the fiber activity induced by the stimulation.

		It first propagates the action potentials (APs) induced by the stimulation along
		the fiber and then it checks whether a new pulse of stimulation occurred.
		In this case an event is sent to all the connected cells at the time = time
		Then, it checks whether a natural AP reached the end of the fiber
		and in this case it sends an event to the connected cells at time = time.
		It then propagates the natural action potentials (APs) along the fiber
		taking in to account possible collision with EES induced AP.

		Keyword arguments:
		time -- current simulation time, necessary for synaptic connections
		"""
		if self._fineTask:
			if self._index < self._run_time/self._step:
				if self._poisson_firings[self._index] == 1: self.cell.EES = 1
				else: self.cell.EES = 0
				self._index += 1

		dt = time-self._oldTime
		self._oldTime = time
		#Propagates the ees antidromic action pontentials
		for i in range(len(self._eesSpikes)):
			if self._eesSpikes[i] != None:
				if self._eesSpikes[i] <= -self._refractoryPeriod:
					self._eesSpikes[i]=None
					if self._debug: print("\t\tAntidromic spike arrived at origin - refPeriod at time: %f" % (time))
				else: self._eesSpikes[i] -= dt

		#Check whether a new pulse of stimulation occured
		if self.cell.EES==1:
			self.cell.EES=0
			#check whether the fiber isn't in refractory period
			if time - self._lastStimPosSpikeTime > self._refractoryPeriod:
				if self._debug: print("\tStimulation pulse occured at time: %f" % (time))
				self._lastStimPosSpikeTime = time
				self._fire.event(time+self._delay-self._stimPosition,1)
				#print(self._lastStimPosSpikeTime)
				self._rasterfire=1
				for i in range(len(self._eesSpikes)):
					if self._eesSpikes[i] == None:
						self._eesSpikes[i] = self._stimPosition
						break #attention if not found AP of EES is not considered - depends on the size of eesSpikes

		# Check whether a natural spike arrived to the end of the fiber
		for i in range(len(self._naturalSpikes)):
			if self._naturalSpikes[i] is not None and self._naturalSpikes[i]>=self._delay-self._tolerance:
				self._fire.event(time,1)
				self._rasterfire=1
				self._lastNaturalSpikeTime_matrix.append(self._lastNaturalSpikeTime)
				self._firetime=time
				self._naturalSpikes[i] = None
				self._nNaturalArrived+=1 # for statistics
				if self._debug: print("\t\t\tnatural spike arrived at time: %f" % (time))

		#update _naturalSpikes
		for i in range(len(self._naturalSpikes)):
			if self._naturalSpikes[i]==None: continue
			#check for collision
			for j in range(len(self._eesSpikes)):
				if self._eesSpikes[j]==None: continue
				if self._naturalSpikes[i]+self.__class__.__updatePeriod > self._eesSpikes[j]-self._tolerance \
				or self._naturalSpikes[i] < self._eesSpikes[j]+self._tolerance:
					self._naturalSpikes[i] = None
					self._eesSpikes[j] = None
					self._nCollisions+=1
					if self._debug: print("\t\t\t\tantidromic collision occurred at time: %f" % (time))
					break
			#advance natural AP
			if self._naturalSpikes[i]!=None:
				self._naturalSpikes[i]+=dt
				if self._naturalSpikes[i] > self._stimPosition-self._tolerance and self._naturalSpikes[i] < self._stimPosition+self._tolerance: #9.5-+0.01
					if time - self._lastStimPosSpikeTime <= self._refractoryPeriod:
						self._naturalSpikes[i]=None
					else:
						self._lastStimPosSpikeTime = time
		#check for new AP
		if (time-self._lastNaturalSpikeTime)>=self._interval-(self.__class__.__updatePeriod/2.):
			self._lastNaturalSpikeTime = time
			for i in range(len(self._naturalSpikes)):
				if self._naturalSpikes[i]==None:
					self._naturalSpikes[i]=0
					self._nNaturalSent+=1
					if self._debug: print("\tsensory spike generated at time: %f" % (time))
					break  #attention if not found, AP of EES is not considered - size of naturalSpike
		
		if self._rasterfire==1:
			event=time/h.dt
			self._rasterlastSpikes.append(event)
			self._rasterfire=0

	def get_delay(self):
		""" Return the time delay in ms needed by a spike to travel the whole fiber. """
		return self._delay

	def set_recording(self, flag, segment):
		""" Set the recording flag and segment.
		This is used to record the affernt natural and ees-induced
		APs in one fiber segment.

		Keyword arguments:
		segment -- fiber segment to record (between 0 and fibers delay)
		time -- current simulation time
		"""
		if segment>self._delay: raise Exception("Segment to record out of limits")
		self._record = flag
		self._segmentToRecord = segment
		self._trigger = []
		self._naturalSignals = []
		self._eesInducedSignals = []
		self._time = []

	def _record_segment(self, time):
		""" Record the fiber segment.  """
		if np.any(np.isclose(0,np.array(self._naturalSpikes,dtype=np.float),rtol=self.__class__.__updatePeriod/4.)):
			self._trigger.append(1)
		else: self._trigger.append(0)

		if np.any(np.isclose(self._segmentToRecord,np.array(self._naturalSpikes,dtype=np.float),rtol=self.__class__.__updatePeriod/4.)):
			self._naturalSignals.append(1)
		else: self._naturalSignals.append(0)

		if np.any(np.isclose(self._segmentToRecord,np.array(self._eesSpikes,dtype=np.float),rtol=self.__class__.__updatePeriod/4.)):
			self._eesInducedSignals.append(1)
		else: self._eesInducedSignals.append(0)

		self._time.append(time)

	def get_recording(self):
		""" Get the recorded signal """
		if self._record: return self._naturalSignals,self._eesInducedSignals,self._trigger,self._time
		else: return None,None,None,None

	def get_stats(self):
		""" Return a touple containing statistics of the fiber after a simulation is performed. """
		if float(self._nNaturalArrived+self._nCollisions)==0: percErasedAp = 0
		else: percErasedAp = float(100*self._nCollisions)/float(self._nNaturalArrived+self._nCollisions)
		return self._nNaturalSent,self._nNaturalArrived,self._nCollisions,percErasedAp

	def create_poisson_firing(self,freq,tStop,integration_step):
		self._run_time = tStop
		self._step=integration_step
		frequency=freq/2000

		T = np.arange(0,self._run_time, self._step,dtype=float)
		Y = np.maximum(frequency*np.sin(T*2*math.pi*.5/1000)+frequency,0.001)

		tf = TimeFunction((T, Y),dt=self._step)

		# We define a 1 dimensional inhomogeneous Poisson process with the
		# intensity function seen above
		in_poi = SimuInhomogeneousPoisson([tf], end_time=self._run_time, verbose=False)

		# We activate intensity tracking and launch simulation
		in_poi.track_intensity(0.1)
		in_poi.simulate()
		numbins=int(self._run_time/self._step)
		idxstart=0
		idxend=self._step
		firing=np.zeros(numbins)
		for bin in range(numbins):
			raster = np.where((in_poi.timestamps[0] >= idxstart) & (in_poi.timestamps[0] <= idxend))
			if len(raster[0])>0:
				firing[bin] =1
			else:
				firing[bin] = 0
			idxstart = idxend+0.000000001
			idxend = idxend+self._step

		self._poisson_firings=firing

	@classmethod
	def get_update_period(cls):
		""" Return the time period between calls of the update fcn. """
		return CSFiber.__updatePeriod

	@classmethod
	def get_ees_weight(cls):
		""" Return the weight of a connection between an ees object and this cell. """
		return CSFiber.__eesWeight

	@classmethod
	def get_max_ees_frequency(cls):
		""" Return the weight of a connection between an ees object and this cell. """
		return CSFiber.__maxEesFrequency


if __name__ == '__main__':

	import sys
	sys.path.append('../code')
	from cells import IntFire
	from simulations import CellsRecording
	import matplotlib.pyplot as plt
	from tools import firings_tools as tlsf


	class AfferentRecording(CellsRecording):
		def __init__(self, parallelContext, cells, modelType, tStop, CSFibers):
			CellsRecording.__init__(self,parallelContext, cells, modelType, tStop)
			self.CSFibers = CSFibers
			self.actionPotentials = []
			self._nc = []
			for af in self.CSFibers:
				self._nc.append(af.connect_to_target(None))
				self.actionPotentials.append(h.Vector())
				self._nc[-1].record(self.actionPotentials[-1])

		def _update(self):
			CellsRecording._update(self)
			if h.t%CSFiber.get_update_period() < self._get_integration_step():
				for af in self.CSFibers:
					af.update(h.t)
			if h.t%100 < self._get_integration_step():
				for af in self.CSFibers:
					af.set_firing_rate(int(h.t/10.-10))
		def _end_integration(self):
			""" Print the total simulation time and extract the results. """
			CellsRecording._end_integration(self)
			self._extract_results()
		def _extract_results(self):
			""" Extract the simulation results. """
			self.firings = tlsf.exctract_firings(self.actionPotentials,self._get_tstop())


	pc = h.ParallelContext()
	simTime = 1000
	cell = IntFire()
	target = cell.cell
	nAfferents = 100
	affFibers = [CSFiber(5) for x in range(nAfferents)]
	nc = []
	for af in affFibers:
			af.set_firing_rate(0)
			nc.append(h.NetCon(af.cell,target))
			nc[-1].weight[0] = 0.001
	cellDict = {"cell":[cell]}
	modelType = {"cell":"artificial"}
	sim = AfferentRecording(pc,cellDict,modelType,simTime,affFibers)
	sim.run()
	# To check whether the fibere fire synchronously or not...
	plt.imshow(sim.firings,interpolation='nearest',origin="lower",aspect='auto')
	plt.show()
	sim.plot("Afferent fiber test - no synchronization after setting the fr")
