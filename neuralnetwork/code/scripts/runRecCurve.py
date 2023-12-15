import argparse
import time
import sys
sys.path.append('../code')
from mpi4py import MPI
from neuron import h
import numpy as np
from tools import general_tools  as gt
from tools import seed_handler as sh

comm = MPI.COMM_WORLD
sizeComm = comm.Get_size()
rank = comm.Get_rank()


def main():
	""" This program launches a ForwardSimulation simulation with a predefined NeuralNetwork structure,
	different stimulation amplitudes are tested to evealuate the muslce recruitment curve.
	The plots resulting from this simulation are saved in the results folder.

	This program can be executed both with and without MPI. In case MPI is used the cells
	of the NeuralNetwork are shared between the different hosts in order to speed up the
	simulation.

	python ./scripts/runRecCurve.py fsMnMod_NER.txt test --muscleName TRI --membranePotential 60 --mnReal 3 --seed 564318


	"""

	parser = argparse.ArgumentParser(
		description="Estimate the reflex responses induced by a range of stimulation amplitudes")
	parser.add_argument("inputFile", help="neural network structure file")
	parser.add_argument("name", help="name to add at the output files", type=str, default="TET")
	parser.add_argument("--mnReal", help=" real mn flag", action="store_true")
	parser.add_argument("--seed",
						help="positive seed used to initialize random number generators (default = time.time())",
						type=int, choices=[gt.Range(0, 999999)])
	parser.add_argument("--membranePotential", help="flag to compute the membrane potential", action="store_true")
	parser.add_argument("--ionChannel", help="flag to compute the ionic current", action="store_true")
	parser.add_argument("--muscleName", help="flag to compute the membrane potential", type=str, default="TRI")
	parser.add_argument("Frequency", help="eesFrequency", type=float, default=10, choices=[gt.Range(0, 1000)])
	parser.add_argument("Amplitude", help="eesAmplitude", type=float, default=5, choices=[gt.Range(0, 10)])
	args = parser.parse_args()

	if args.seed is not None:
		sh.save_seed(args.seed)
		n = args.seed
	else:
		n = int(time.time())
		sh.save_seed(n)

	# Import simulation specific modules
	from simulations import ForwardSimulation
	from simulations import ForSimSpinalModulation
	from NeuralNetwork import NeuralNetwork
	from EES import EES
	from CST import CST

	# Initialize parameters
	eesAmplitudes = [[x, 0, 0] for x in np.arange(args.Amplitude / 10, 1.1, .1)]
	simTime = 500

	# Create a Neuron ParallelContext object to support parallel simulations
	pc = h.ParallelContext()
	nn = NeuralNetwork(pc, args.inputFile)

	if 'CM' in nn.get_corticomotoneurons_names():
			ICfreq = 60
			ICstimAmplitude = [[0, x, 0] for x in np.arange(0.01, 1.04, 2)]
			ICstim = []
			idxid = 0
			for muscle in nn.cells:
				for cellName in nn.cells[muscle]:
					if cellName in nn._corticoMotoneuronsNames:
						for cell in nn.cells[muscle][cellName]:
							ICstim.append(CST(pc, nn, ICstimAmplitude[0], ICfreq, idxid))
							idxid += 1
			afferentsInput = None
	else:
		ICstim = None
		ICstimAmplitude = [[0, x, 0] for x in np.arange(0, 1, 2)]
		ICfreq = 10000000

	if 'Iaf' in nn.get_primary_afferents_names():
		ees = EES(pc, nn, eesAmplitudes[0], args.Frequency)
		npulses = 1
		afferentsInput = None
	else:
		ees = None
		eesAmplitudes = [[0, x, 0] for x in np.arange(0, 1, 2)]
		npulses = 0

	program = ['python3', './scripts/runRecCurve.py', args.inputFile, "name", "--muscleName", args.muscleName,
			   "--mnReal", args.mnReal, "SCSfreq", str(args.Frequency), \
			   args.name, "--membranePotential", args.membranePotential, "--ionChannel", args.ionChannel,\
			   "--seed", str(n)]

	infocode = ["SCS", str(ees), "SCSamplitude", str(eesAmplitudes), "nPulsesSCS", str(npulses), \
				"ICstim", str(ICstim), "ICamplitude", str(ICstimAmplitude), "ICfrequency", str(ICfreq)]

	eesModulation = None

	if args.membranePotential:
		if args.mnReal:
			cellsToRecord = {"MnReal": [mn.soma for mn in nn.cells[args.muscleName]['MnReal']]}
			modelTypes = {"MnReal": "real"}
		else:
			cellsToRecord = {"Mn": nn.cells[args.muscleName]['Mn']}
			modelTypes = {"Mn": "artificial"}
			nn._drug = False
		simulation = ForSimSpinalModulation(pc, eesAmplitudes[0], nn, cellsToRecord, modelTypes, ICfreq,
											afferentsInput, ICstim, eesModulation, simTime)
	else:
		simulation = ForwardSimulation(pc, nn, ICfreq, afferentsInput, ees, eesModulation, simTime)

	for eesAmplitude in eesAmplitudes:
		if ees != None:
			ees.set_amplitude(eesAmplitude)
		if ICstim != None:
			iid = 0
			for muscle in nn.cells:
				for cellName in nn.cells[muscle]:
					if cellName in nn._corticoMotoneuronsNames:
						for i in nn.cells[muscle][cellName]:
							ICstim[iid].set_amplitude(eesAmplitude)
							iid += 1
		simulation.run()
		if ees != None and args.mnReal:
			[ees_times, ees_pulses] = ees.get_pulses(simTime)
		else:
			ees_pulses = 0
		if ICstim != None:
			iid = 0
			ic_pulses = []
			for muscle in nn.cells:
				for cell in nn.cells[muscle][simulation._CM]:
					[ic_times, ic_pulses_all] = ICstim[iid].get_pulses(simTime)
					ic_pulses.append(ic_times)
					iid += 1
		else:
			ic_pulses = 0
		simulation.get_results(nn, args.muscleName, ees_pulses, args.name, eesAmplitude, args.Frequency,
							   ICfreq, ICstimAmplitude, simTime, program, infocode, ic_pulses, args.mnReal)
		fileName = "%.2s_amp_%f_freq_%.2f" % (args.name, eesAmplitude[0], args.Frequency)
		if args.membranePotential:
			simulation.plot_membrane_potatial(fileName)
		if args.ionChannel:
			simulation.plot_ion_channel(fileName)

if __name__ == '__main__':
	main()
