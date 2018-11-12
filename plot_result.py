import sys
import numpy as np

import cantera as ct
import matplotlib.pyplot as plt

gas = ct.Solution('gri30.cti')
temp = 1200.0
pres = 5*ct.one_atm

end_time = 3e-3
step_time = 1e-5
threshold = 80


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
for t in np.arange(0, end_time, step_time):
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

max_delta = 0
index_temp = 0
for i in range(len(states.T)-2):
    if(states.T[i+1] - states.T[i] > max_delta):
        max_delta = states.T[i+1] - states.T[i]
        index_temp = i

ignite_time_precise = float(states.t[index_temp])
precise_T   = float(states.T[-1]        )
precise_CH4 = float(states('CH4').X[-1] )
precise_O2  = float(states('O2').X[-1]  )
precise_CO2 = float(states('CO2').X[-1] )
precise_H2O = float(states('H2O').X[-1] )
precise_H   = float(states('H').X[-1]   )
precise_OH  = float(states('OH').X[-1]  )

plt.subplot(2,2,1)
plt.plot(states.t, states.T, label='precise mechanism')
plt.xlabel('Time (ms)')
plt.ylabel('Temperature (K)')
plt.subplot(2,2,2)
plt.plot(states.t, states('CH4').X, label='precise mechanism')
plt.xlabel('Time (ms)')
plt.ylabel('CH4 Mole Fraction')
plt.subplot(2,2,3)
plt.plot(states.t, states('O2').X, label='precise mechanism')
plt.xlabel('Time (ms)')
plt.ylabel('O2 Mole Fraction')
plt.subplot(2,2,4)
plt.plot(states.t, states('CO2').X, label='precise mechanism')
plt.xlabel('Time (ms)')
plt.ylabel('CO2 Mole Fraction')
plt.tight_layout()

maxs = [max(map(abs,x)) for x in B]

R = [x[1] for x in zip(maxs, gas.reactions()) if x[0] > threshold]
print(len(R))
gas2 = ct.Solution(thermo='IdealGas', kinetics='GasKinetics',species=gas.species(),reactions=R)
gas2.TPX = temp, pres, 'CH4:1, O2:2'
r = ct.IdealGasConstPressureReactor(gas2)
sim = ct.ReactorNet([r])
states = ct.SolutionArray(gas2, extra=['t'])

for t in np.arange(0, end_time, step_time):
    sim.advance(t)
    states.append(r.thermo.state, t=1000*t)

max_delta = 0
for i in range(len(states.T)-2):
    if(states.T[i+1] - states.T[i] > max_delta):
        max_delta = states.T[i+1] - states.T[i]
        index_temp = i
ignite_time = states.t[index_temp]
print("after reduction, ignite_time: ",ignite_time,'Tend',states.T[-1])

plt.subplot(2,2,1)
plt.plot(states.t, states.T,'--',label='Redeuced to {} reactions'.format(gas2.n_reactions))
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Temperature (K)')
plt.subplot(2,2,2)
plt.plot(states.t, states('CH4').X,'--',label='Redeuced to {} reactions'.format(gas2.n_reactions))
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('CH4 Mole Fraction')
plt.subplot(2,2,3)
plt.plot(states.t, states('O2').X,'--',label='Redeuced to {} reactions'.format(gas2.n_reactions))
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('O2 Mole Fraction')
plt.subplot(2,2,4)
plt.plot(states.t, states('CO2').X,'--',label='Redeuced to {} reactions'.format(gas2.n_reactions))
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('CO2 Mole Fraction')



import best_fit
tmp_reactions = R.copy()
tmp_parameters = best_fit.best_fit[-1][1]
print(len(tmp_reactions))
for i in range(len(tmp_reactions)):
    if tmp_reactions[i].reaction_type !=4:
        tmp_reactions[i].rate = ct.Arrhenius(A = tmp_parameters[3*i], b = tmp_parameters[3*i+1], E = tmp_parameters[3*i+2])
    else:
        tmp_reactions[i].low_rate = ct.Arrhenius(A = tmp_parameters[3*i], b = tmp_parameters[3*i+1], E = tmp_parameters[3*i+2])

gas2 = ct.Solution(thermo='IdealGas', kinetics='GasKinetics',species=gas.species(),reactions=tmp_reactions)

gas2.TPX = temp, pres, 'CH4:1, O2:2'
r = ct.IdealGasConstPressureReactor(gas2)
sim = ct.ReactorNet([r])
states = ct.SolutionArray(gas2, extra=['t'])
for t in np.arange(0, end_time, step_time):
    sim.advance(t)
    states.append(r.thermo.state, t=1000*t)

max_delta = 0
index_temp = 0
for i in range(len(states.T)-2):
    if(states.T[i+1] - states.T[i] > max_delta):
        max_delta = states.T[i+1] - states.T[i]
        index_temp = i
        
ignite_time_optimised = float(states.t[index_temp])
optimised_T   = float(states.T[-1]        )
optimised_CH4 = float(states('CH4').X[-1] )
optimised_O2  = float(states('O2').X[-1]  )
optimised_CO2 = float(states('CO2').X[-1] )
optimised_H2O = float(states('H2O').X[-1] )
optimised_H   = float(states('H').X[-1]   )
optimised_OH  = float(states('OH').X[-1]  )
print('optimized', ignite_time_optimised, states.T[-1])

print("item    {:>10s} {:>10s} {:>10s}".format("precise","optimised","different"))
print("ig_t    {:9.4e} {:9.4e} {:9.4e}".format(ignite_time_precise,  ignite_time_optimised,   abs(ignite_time_optimised - ignite_time_precise)))
print("T       {:9.4e} {:9.4e} {:9.4e}".format(precise_T,            optimised_T,             abs(optimised_T   - precise_T)                  ))
print("CH4     {:9.4e} {:9.4e} {:9.4e}".format(precise_CH4,          optimised_CH4,           abs(optimised_CH4 - precise_CH4)                ))
print("O2      {:9.4e} {:9.4e} {:9.4e}".format(precise_O2,           optimised_O2,            abs(optimised_O2  - precise_O2)                 ))
print("CO2     {:9.4e} {:9.4e} {:9.4e}".format(precise_CO2,          optimised_CO2,           abs(optimised_CO2 - precise_CO2)                ))
print("H2O     {:9.4e} {:9.4e} {:9.4e}".format(precise_H2O,          optimised_H2O,           abs(optimised_H2O - precise_H2O)                ))
print("H       {:9.4e} {:9.4e} {:9.4e}".format(precise_H,            optimised_H,             abs(optimised_H   - precise_H)                   ))
print("OH      {:9.4e} {:9.4e} {:9.4e}".format(precise_OH,           optimised_OH,            abs(optimised_OH  - precise_OH)                 ))

plt.subplot(2,2,1)
plt.plot(states.t, states.T,'-.',label= 'optimized')
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Temperature (K)')
plt.subplot(2,2,2)
plt.plot(states.t, states('CH4').X,'-.',label='optimized')
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('CH4 Mole Fraction')
plt.subplot(2,2,3)
plt.plot(states.t, states('O2').X,'-.',label='optimized')
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('O2 Mole Fraction')
plt.subplot(2,2,4)
plt.plot(states.t, states('CO2').X,'-.',label='optimized')
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('CO2 Mole Fraction')
plt.tight_layout()
plt.show()
