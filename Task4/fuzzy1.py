from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from tank import Tank

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


client = RemoteAPIClient()
sim = client.require('sim')

sim.setStepping(True)

my_tank = Tank(one_sensor=True)

# Define the fuzzy variables
distance = ctrl.Antecedent(np.arange(0, 10, .01), 'distance')
velocity = ctrl.Consequent(np.arange(0, 10, .1), 'velocity')

# Define membership functions for distance
distance['very_close'] = fuzz.trimf(distance.universe, [0, 0, 0.4])
distance['close'] = fuzz.trimf(distance.universe, [0.4, 1, 1])
distance['medium'] = fuzz.trimf(distance.universe, [1, 3, 5])
distance['far'] = fuzz.trimf(distance.universe, [3, 5, 10])

# Define membership functions for velocity
velocity['stop'] = fuzz.trimf(velocity.universe, [0, 0, 0])
velocity['slow'] = fuzz.trimf(velocity.universe, [0, 1, 3])
velocity['moderate'] = fuzz.trimf(velocity.universe, [1, 3, 5])
velocity['fast'] = fuzz.trimf(velocity.universe, [3, 5, 10])

# Define the fuzzy rules
rule1 = ctrl.Rule(distance['very_close'], velocity['stop'])
rule2 = ctrl.Rule(distance['close'], velocity['slow'])
rule3 = ctrl.Rule(distance['medium'], velocity['moderate'])
rule4 = ctrl.Rule(distance['far'], velocity['fast'])

# Create a control system and simulation
velocity_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
velocity_sim = ctrl.ControlSystemSimulation(velocity_ctrl)

sim.startSimulation()
my_tank.stop()
while (t := sim.getSimulationTime()) < 35:
    dp = my_tank.read_proximity_sensors()['N']['detectedPoint']

    # Simulate the system
    velocity_sim.input['distance'] = float(dp)
    velocity_sim.compute()
    vel = velocity_sim.output['velocity']


    my_tank.forward(vel)
    sim.step()

sim.stopSimulation()