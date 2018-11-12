
import sys
import numpy as np

import cantera as ct
import matplotlib.pyplot as plt

gas = ct.Solution('gri30.cti')
temp = 1200.0
pres = ct.one_atm

gas.TPX = temp, pres, 'CH4:1, O2:2'
r = ct.IdealGasConstPressureReactor(gas, name='R1')
sim = ct.ReactorNet([r])
for i in range(gas.n_reactions):
    r.add_sensitivity_reaction(i)

# set the tolerances for the solution and for the sensitivity coefficients
sim.rtol = 1.0e-6
sim.atol = 1.0e-15
sim.rtol_sensitivity = 1.0e-6
sim.atol_sensitivity = 1.0e-6

states = ct.SolutionArray(gas, extra=['t'])


sCH4 = []
sO2 = []
sCO2 = []
sH2O = []
sH = []
sOH = []
j = 0
tt0=[]
TT0=[]
for t in np.arange(0, 1.5e-2, 5e-5):
    sim.advance(t)
#    s2 = sim.sensitivity('OH', 2) # sensitivity of OH to reaction 2
#    s3 = sim.sensitivity('OH', 3) # sensitivity of OH to reaction 3
    states.append(r.thermo.state, t=1000*t)
    sCH4.append([])
    sO2.append([])
    sCO2.append([])
    sH2O.append([])
    sH.append([])
    sOH.append([])
    tt0.append(1000*t)
    TT0.append(r.T)
    for i in range(gas.n_reactions):
        sCH4[j].append(sim.sensitivity('CH4', i)) 
        sO2[j].append(sim.sensitivity('O2', i))
        sCO2[j].append(sim.sensitivity('CO2', i)) 
        sH2O[j].append(sim.sensitivity('H2O', i)) 
        sH[j].append(sim.sensitivity('H', i))
        sOH[j].append(sim.sensitivity('OH', i))
    j = j+1
sCH4 = list(map(list, zip(*sCH4)))
sO2 = list(map(list, zip(*sO2)))
sCO2 = list(map(list, zip(*sCO2)))
sH2O = list(map(list, zip(*sH2O)))
sH = list(map(list, zip(*sH)))
sOH = list(map(list, zip(*sOH)))
B = [list(map(lambda x,y,z,w,x2,y2:x**2+y**2+z**2+w**2+x2**2+y2**2,x,y,z,w,x2,y2)) for x,y,z,w,x2,y2 in zip(sCH4,sO2,sCO2,sH2O,sH,sOH)]

maxs = [max(map(abs,x)) for x in B]

threshold = 5
R = [x[1] for x in zip(maxs, gas.reactions()) if x[0] > threshold]
# fitness = [x[0] for x in zip(maxs, gas.reactions()) if x[0] > threshold]
# for i,x in enumerate(R):
#     print(i, x)
gas2 = ct.Solution(thermo='IdealGas', kinetics='GasKinetics',species=gas.species(),reactions=R)
gas2.TPX = temp, pres, 'CH4:1, O2:2'
r = ct.IdealGasConstPressureReactor(gas2)
sim = ct.ReactorNet([r])
states = ct.SolutionArray(gas2, extra=['t'])
Arrhenius_parameters = []
for i in range(gas2.n_reactions):
    if gas2.reaction(i).reaction_type !=4:
        Arrhenius_parameters.append(gas2.reaction(i).rate)
    else:
        Arrhenius_parameters.append(gas2.reaction(i).low_rate)
    
    print(Arrhenius_parameters[i])