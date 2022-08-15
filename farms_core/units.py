"""Units"""

from .options import Options


class SimulationUnitScaling(Options):
    """Simulation scaling

    1 [m] in reality = self.meters [m] in simulation
    1 [s] in reality = self.seconds [s] in simulation
    1 [kg] in reality = self.kilograms [kg] in simulation

    """

    def __init__(
            self,
            meters: float = 1,
            seconds: float = 1,
            kilograms: float = 1,
    ):
        super().__init__()
        self.meters = meters
        self.seconds = seconds
        self.kilograms = kilograms

    @property
    def hertz(self) -> float:
        """Hertz (frequency)

        Scaled as self.hertz = 1/self.seconds

        """
        return 1./self.seconds

    @property
    def newtons(self) -> float:
        """Newtons

        Scaled as self.newtons = self.kilograms*self.meters/self.time**2

        """
        return self.kilograms*self.acceleration

    @property
    def torques(self) -> float:
        """Torques

        Scaled as self.torques = self.kilograms*self.meters**2/self.time**2

        """
        return self.newtons*self.meters

    @property
    def velocity(self) -> float:
        """Velocity

        Scaled as self.velocities = self.meters/self.seconds

        """
        return self.meters/self.seconds

    @property
    def angular_velocity(self) -> float:
        """Angular velocity

        Scaled as self.angular_velocities = 1/self.seconds

        """
        return 1/self.seconds

    @property
    def acceleration(self) -> float:
        """Acceleration

        Scaled as self.gravity = self.meters/self.seconds**2

        """
        return self.velocity/self.seconds

    @property
    def gravity(self) -> float:
        """Gravity

        Scaled as self.gravity = self.meters/self.seconds**2

        """
        return self.acceleration

    @property
    def volume(self) -> float:
        """Volume

        Scaled as self.volume = self.meters**3

        """
        return self.meters**3

    @property
    def density(self) -> float:
        """Density

        Scaled as self.density = self.kilograms/self.meters**3

        """
        return self.kilograms/self.volume

    @property
    def stiffness(self) -> float:
        """Stiffness

        Scaled as self.stiffness = self.netwons/self.meters

        """
        return self.newtons/self.meters

    @property
    def damping(self) -> float:
        """Damping

        Scaled as self.damping = self.netwons/self.velocity

        """
        return self.newtons/self.velocity

    @property
    def angular_stiffness(self) -> float:
        """Angular stiffness

        Scaled as self.angular_stiffness = self.torques/radian

        """
        return self.torques

    @property
    def angular_damping(self) -> float:
        """Angular damping

        Scaled as self.angular_damping = self.torques/self.angular_velocity

        """
        return self.torques/self.angular_velocity

    @property
    def inertia(self) -> float:
        """Inertia

        Scaled as self.inertia = self.kilograms*self.meters**2

        """
        return self.kilograms*self.meters**2
