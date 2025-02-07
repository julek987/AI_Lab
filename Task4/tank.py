from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class Tank:
    def __init__(self, one_sensor=False):
        # Connect to the remote API client
        print('ceated a tank!')
        client = RemoteAPIClient()
        self.sim = client.getObject('sim')

        # Definde if all sensors will be used
        self.one_sensor = one_sensor

        # Get handles to robot drivers
        self.left_front_handle = self.sim.getObject('/left_front')
        self.left_back_handle = self.sim.getObject('/left_back')
        self.right_back_handle = self.sim.getObject('/right_back')
        self.right_front_handle = self.sim.getObject('/right_front')

        self.side_handles = []
        for l in 'rl':
            for i in range(1, 7):
                handle = self.sim.getObject(f'/sj_{l}_{i}')
                self.side_handles.append(handle)

        # Initial velocity
        self.leftvelocity = 0
        self.rightvelocity = 0
        self.MaxVel = 10
        self.dVel = 1
        self.FORCE = 100

        # Proximity sensors
        self.proximity_sensors = ["EN", "ES", "NE", "NW", "SE", "SW", "WN", "WS"] if not self.one_sensor else ['N']
        self.proximity_sensors_handles = []

        # Get handles to proximity sensors
        for sensor in self.proximity_sensors:
            handle = self.sim.getObject(f'/Proximity_sensor_{sensor}' if not self.one_sensor else f'/Proximity_sensor')
            self.proximity_sensors_handles.append(handle)

    def stop(self):
        # Set drivers to stop mode
        force = 0
        self.sim.setJointForce(self.left_front_handle, force)
        self.sim.setJointForce(self.left_back_handle, force)
        self.sim.setJointForce(self.right_back_handle, force)
        self.sim.setJointForce(self.right_front_handle, force)

        force = self.FORCE
        for h in self.side_handles:
            self.sim.setJointForce(h, force)

        # Brake
        self.leftvelocity = 10
        self.rightvelocity = 10
        self.sim.setJointTargetVelocity(self.left_front_handle, self.leftvelocity)
        self.sim.setJointTargetVelocity(self.left_back_handle, self.leftvelocity)
        self.sim.setJointTargetVelocity(self.right_back_handle, self.rightvelocity)
        self.sim.setJointTargetVelocity(self.right_front_handle, self.rightvelocity)

    def go(self):
        # Set drivers to go mode
        force = self.FORCE
        self.sim.setJointForce(self.left_front_handle, force)
        self.sim.setJointForce(self.left_back_handle, force)
        self.sim.setJointForce(self.right_back_handle, force)
        self.sim.setJointForce(self.right_front_handle, force)

        force = 0
        for h in self.side_handles:
            self.sim.setJointForce(h, force)

    def setVelocity(self):
        # Verify if the velocity is in the correct range
        self.leftvelocity = max(min(self.leftvelocity, self.MaxVel), -self.MaxVel)
        self.rightvelocity = max(min(self.rightvelocity, self.MaxVel), -self.MaxVel)

        # Send the velocity values to the drivers
        self.sim.setJointTargetVelocity(self.left_back_handle, self.leftvelocity)
        self.sim.setJointTargetVelocity(self.right_back_handle, self.rightvelocity)

    def forward(self, velocity=None):
        self.go()
        if velocity is not None:
            self.leftvelocity = velocity
            self.rightvelocity = velocity
        else:
            self.rightvelocity = self.leftvelocity = (self.leftvelocity + self.rightvelocity) / 2
            self.leftvelocity += self.dVel
            self.rightvelocity += self.dVel
        self.setVelocity()

    def backward(self, velocity=None):
        self.go()
        if velocity is not None:
            self.leftvelocity = -velocity
            self.rightvelocity = -velocity
        else:
            self.rightvelocity = self.leftvelocity = (self.leftvelocity + self.rightvelocity) / 2
            self.leftvelocity -= self.dVel
            self.rightvelocity -= self.dVel
        self.setVelocity()

    def turn_left(self, velocity=None):
        self.go()
        if velocity is not None:
            self.leftvelocity = -velocity
            self.rightvelocity = velocity
        else:
            self.leftvelocity -= self.dVel
            self.rightvelocity += self.dVel
        self.setVelocity()

    def turn_right(self, velocity=None):
        self.go()
        if velocity is not None:
            self.leftvelocity = velocity
            self.rightvelocity = -velocity
        else:
            self.leftvelocity += self.dVel
            self.rightvelocity -= self.dVel
        self.setVelocity()

    def read_proximity_sensors(self):
        # Read and print values from proximity sensors
        sensor_data = {}
        for sensor_name, sensor_handle in zip(self.proximity_sensors, self.proximity_sensors_handles):
            result = self.sim.readProximitySensor(sensor_handle)
            detectionState = result[0]
            detectedPoint = result[1]
            detectedObjectHandle = result[2]
            detectedSurfaceNormalVector = result[3]
            sensor_data[sensor_name] = {
                "detectionState": detectionState,
                "detectedPoint": detectedPoint,
                "detectedObjectHandle": detectedObjectHandle,
                "detectedSurfaceNormalVector": detectedSurfaceNormalVector
            }
        return sensor_data