import sys
import numpy as np

import cantera as ct
import matplotlib.pyplot as plt

end_time = 3e-3
step_time = 1e-5
threshold = 50

gas = ct.Solution('gri30.cti')
temp = 1200.0
pres = 5*ct.one_atm

gas.TPX = temp, pres, 'CH4:1, O2:2'
r = ct.IdealGasConstPressureReactor(gas, name='R1')
sim = ct.ReactorNet([r])

# enable sensitivity with respect to the rates of the first 10
# reactions (reactions 0 through 9)
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
for i in range(len(states.T)-1):
    if(states.T[i+1] - states.T[i] > max_delta):
        max_delta = states.T[i+1] - states.T[i]
        index_temp = i
ignite_time_precise = states.t[index_temp]

precise_T   = float(states.T[-1]        )
precise_CH4 = float(states('CH4').X[-1] )
precise_O2  = float(states('O2').X[-1]  )
precise_CO2 = float(states('CO2').X[-1] )
precise_H2O = float(states('H2O').X[-1] )
precise_H   = float(states('H').X[-1]   )
precise_OH  = float(states('OH').X[-1]  )


maxs = [max(map(abs,x)) for x in B]


R = [x[1] for x in zip(maxs, gas.reactions()) if x[0] > threshold]
gas2 = ct.Solution(thermo='IdealGas', kinetics='GasKinetics',species=gas.species(),reactions=R)
gas2.TPX = temp, pres, 'CH4:1, O2:2'
r = ct.IdealGasConstPressureReactor(gas2)
sim = ct.ReactorNet([r])
states = ct.SolutionArray(gas2, extra=['t'])


for t in np.arange(0, end_time, step_time):
    sim.advance(t)
    states.append(r.thermo.state, t=1000*t)

reduced_T   = float(states.T[-1]        )
reduced_CH4 = float(states('CH4').X[-1] )
reduced_O2  = float(states('O2').X[-1]  )
reduced_CO2 = float(states('CO2').X[-1] )
reduced_H2O = float(states('H2O').X[-1] )
reduced_H   = float(states('H').X[-1]   )
reduced_OH  = float(states('OH').X[-1]  )

max_delta = 0
for i in range(len(states.T)-1):
    if(states.T[i+1] - states.T[i] > max_delta):
        max_delta = states.T[i+1] - states.T[i]
        index_temp = i
ignite_time_reduced = states.t[index_temp]

Arrhenius_parameters = []
for i in range(gas2.n_reactions):
    if gas2.reaction(i).reaction_type !=4:
        Arrhenius_parameters.append(gas2.reaction(i).rate)
    else:
        Arrhenius_parameters.append(gas2.reaction(i).low_rate)

print(len(R))
print("item    {:>9s} {:>9s} {:>9s}".format("precise","reduced","diff"))
print("ig_t    {:9.4e} {:9.4e} {:9.4e}".format(ignite_time_precise,  ignite_time_reduced,   abs(ignite_time_reduced - ignite_time_precise)))
print("T       {:9.4e} {:9.4e} {:9.4e}".format(precise_T,            reduced_T,             abs(precise_T - reduced_T)                    ))
print("CH4     {:9.4e} {:9.4e} {:9.4e}".format(precise_CH4,          reduced_CH4,           abs(precise_CH4 - reduced_CH4)                ))
print("O2      {:9.4e} {:9.4e} {:9.4e}".format(precise_O2,           reduced_O2,            abs(precise_O2 - reduced_O2)                  ))
print("CO2     {:9.4e} {:9.4e} {:9.4e}".format(precise_CO2,          reduced_CO2,           abs(precise_CO2 - reduced_CO2)                ))
print("H2O     {:9.4e} {:9.4e} {:9.4e}".format(precise_H2O,          reduced_H2O,           abs(precise_H2O - reduced_H2O)                ))
print("H       {:9.4e} {:9.4e} {:9.4e}".format(precise_H,            reduced_H,             abs(precise_H - reduced_H)                    ))
print("OH      {:9.4e} {:9.4e} {:9.4e}".format(precise_OH,           reduced_OH,            abs(precise_OH - reduced_OH)                  ))



'''
from gaft import GAEngine
from gaft.components import DecimalIndividual
from gaft.components import Population
from gaft.operators import TournamentSelection
from gaft.operators import UniformCrossover
from gaft.operators import FlipBitBigMutation

# Built-in best fitness analysis.
from gaft.analysis.fitness_store import FitnessStore
from gaft.analysis.console_output import ConsoleOutput

# Define population.
parameter_range=[]
eps_number = []
for i in range(gas2.n_reactions):
    parameter_range.append((Arrhenius_parameters[i].pre_exponential_factor - abs(Arrhenius_parameters[i].pre_exponential_factor)*0.2 ,
     Arrhenius_parameters[i].pre_exponential_factor + abs(Arrhenius_parameters[i].pre_exponential_factor)*0.2 + 0.1))
    eps_number.append(1e-6)

    parameter_range.append((Arrhenius_parameters[i].temperature_exponent - abs(Arrhenius_parameters[i].temperature_exponent)*0.2 - 0.01,
    Arrhenius_parameters[i].temperature_exponent + abs(Arrhenius_parameters[i].temperature_exponent)*0.2 + 0.01))
    eps_number.append(1e-5)

    parameter_range.append((Arrhenius_parameters[i].activation_energy-abs(Arrhenius_parameters[i].activation_energy)*0.2 - 300,
    Arrhenius_parameters[i].activation_energy + abs(Arrhenius_parameters[i].activation_energy)*0.2 + 300))
    eps_number.append(1e-5)

indv_template = DecimalIndividual(ranges=parameter_range, eps = eps_number)
population = Population(indv_template=indv_template, size=50).init()

# Create genetic operators.
#selection = RouletteWheelSelection()
selection = TournamentSelection()
crossover = UniformCrossover(pc=0.8, pe=0.5)
mutation = FlipBitBigMutation(pm=0.1, pbm=0.55, alpha=0.6)

# Create genetic algorithm engine.
# Here we pass all built-in analysis to engine constructor.
engine = GAEngine(population=population, selection=selection,
                  crossover=crossover, mutation=mutation,
                  analysis=[ConsoleOutput, FitnessStore])

# Define fitness function.
@engine.fitness_register
@engine.minimize
def fitness(indv):
    tmp_parameters = indv.solution
    
    tmp_reactions = R.copy()
    for i in range(len(tmp_reactions)):
        if tmp_reactions[i].reaction_type !=4:
            tmp_reactions[i].rate = ct.Arrhenius(A = tmp_parameters[3*i], \
            b = tmp_parameters[3*i+1], E = tmp_parameters[3*i+2])
        else:
            tmp_reactions[i].low_rate = ct.Arrhenius(A = tmp_parameters[3*i], \
            b = tmp_parameters[3*i+1], E = tmp_parameters[3*i+2])
    gas2 = ct.Solution(thermo='IdealGas', kinetics='GasKinetics',species=gas.species(),reactions=tmp_reactions)

    gas2.TPX = temp, pres, 'CH4:1, O2:2'
    r = ct.IdealGasConstPressureReactor(gas2)
    sim = ct.ReactorNet([r])
    states = ct.SolutionArray(gas2, extra=['t'])
    try:
        for t in np.arange(0, end_time, step_time):
            sim.advance(t)
            states.append(r.thermo.state, t=1000*t)
    except:
        return -1e8

    max_delta = 0
    index_temp = 0
    for i in range(len(states.T)-1):
        if(states.T[i+1] - states.T[i] > max_delta):
            max_delta = states.T[i+1] - states.T[i]
            index_temp = i

    ignite_time_optimised = states.t[index_temp]
    optimised_T = states.T[-1]
    optimised_CH4 = states('CH4').X[-1]
    optimised_O2 = states('O2').X[-1]
    optimised_CO2 = states('CO2').X[-1]
    optimised_H2O = states('H2O').X[-1]
    optimised_H = states('H').X[-1]
    optimised_OH = states('OH').X[-1]

    return float(((ignite_time_optimised - ignite_time_precise)/(abs(ignite_time_reduced - ignite_time_precise) + 0.001))**2\
    + ((optimised_T - precise_T)/(abs(reduced_T - precise_T) + 0.001))**2\
    # + ((optimised_CH4 - precise_CH4)/(abs(reduced_CH4 - precise_CH4) + 0.001))**2\
    # + ((optimised_O2 - precise_O2)/(abs(reduced_O2 - precise_O2) + 0.001))**2\
    # + ((optimised_CO2 - precise_CO2)/(abs(reduced_CO2 - precise_CO2) + 0.001))**2\
    # + ((optimised_H2O - precise_H2O)/(abs(reduced_H2O - precise_H2O) + 0.001))**2\
    # + ((optimised_H - precise_H)/(abs(reduced_H - precise_H) + 0.001))**2\
    # + ((optimised_OH - precise_OH)/(abs(reduced_OH - precise_OH) + 0.001))**2\
    )


if '__main__' == __name__:
    engine.run(ng=300)

'''