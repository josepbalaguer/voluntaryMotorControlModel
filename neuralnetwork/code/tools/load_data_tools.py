import pandas as pd
from tools import general_tools  as gt

def readCsvGeneral(file2read,headerLines,outLabels,signalsName,sep='\t'):
    outDict = {}
    dataFrame = pd.read_csv(file2read,header=headerLines,sep=sep)
    for label,muscle in zip(outLabels,signalsName):
        outDict[label] = dataFrame[muscle].values
    return outDict

def load_afferent_input(nn, eesAmplitude, muscleName, species='monkey' , muscles=None, exp="locomotion"):
    """ Load previously computed afferent inputs"""
    afferentsInput = None
    if species == 'rat':
        muscles = {"ext":"GM","flex":"TA"}
        afferents = {}
        afferents[muscles["flex"]] = {}
        afferents[muscles["ext"]] = {}
        if exp == "locomotion":
            afferents[muscles["flex"]]['Iaf'] = list(gt.load_txt_mpi('../inputFiles/meanFr_Ia_TA_rat.txt'))
            afferents[muscles["flex"]]['IIf'] = list(gt.load_txt_mpi('../inputFiles/meanFr_II_TA_rat.txt'))
            afferents[muscles["ext"]]['Iaf'] = list(gt.load_txt_mpi('../inputFiles/meanFr_Ia_GM_rat.txt'))
            afferents[muscles["ext"]]['IIf'] = list(gt.load_txt_mpi('../inputFiles/meanFr_II_GM_rat.txt'))
        dtUpdateAfferent = 5
        afferentsInput = [afferents,dtUpdateAfferent]
    elif species == 'human':
        muscles = {"ext":"SOL","flex":"TA"}
        afferents = {}
        afferents[muscles["flex"]] = {}
        afferents[muscles["ext"]] = {}
        if exp == "locomotion":
            afferents[muscles["flex"]]['Iaf'] = list(gt.load_txt_mpi("../inputFiles/meanFr_Ia_TA_human.txt"))
            afferents[muscles["flex"]]['IIf'] = list(gt.load_txt_mpi("../inputFiles/meanFr_II_TA_human.txt"))
            afferents[muscles["ext"]]['Iaf'] = list(gt.load_txt_mpi("../inputFiles/meanFr_Ia_SOL_human.txt"))
            afferents[muscles["ext"]]['IIf'] = list(gt.load_txt_mpi("../inputFiles/meanFr_II_SOL_human.txt"))
        dtUpdateAfferent = 5
        afferentsInput = [afferents,dtUpdateAfferent]
    elif species == 'monkey':
        muscles = {"ext": "TRI", "flex": "BIC"}
        afferents = {}
        afferents[muscles["flex"]] = {}
        afferents[muscles["ext"]] = {}
        k=1
        if exp == "locomotion":
            if eesAmplitude[-1][0] == 0 and 'Iaf' in nn.get_primary_afferents_names():
                if 'TRI' in nn.cells:
                    afferents[muscles["ext"]]['Iaf'] = list(gt.load_txt_mpi("../inputFiles/meanFr_Ia_SOL_human.txt"))
                    afferents[muscles["ext"]]['Iaf'] = [x*k for x in afferents[muscles["ext"]]['Iaf']]
                if 'BIC' in nn.cells:
                    afferents[muscles["flex"]]['Iaf'] = list(gt.load_txt_mpi("../inputFiles/meanFr_Ia_BIC_monkey.txt"))
                    afferents[muscles["flex"]]['Iaf'] = [x * k for x in afferents[muscles["flex"]]['Iaf']]

        dtUpdateAfferent = 5
        afferentsInput = [afferents, dtUpdateAfferent]
    return afferentsInput
