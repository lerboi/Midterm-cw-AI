import genome 
from xml.dom.minidom import getDOMImplementation
from enum import Enum
import numpy as np

class MotorType(Enum):
    PULSE = 1
    SINE = 2

class Motor:
    def __init__(self, control_waveform, control_amp, control_freq):
        if control_waveform <= 0.5:
            self.motor_type = MotorType.PULSE
        else:
            self.motor_type = MotorType.SINE
        # Amplitude is kept constant at 1.0 for fair testing across all creatures
        # The control_amp gene is preserved in DNA structure but not applied
        self.amp = 1.0
        self.freq = control_freq
        self.phase = 0


    def get_output(self):
        self.phase = (self.phase + self.freq) % (np.pi * 2)
        if self.motor_type == MotorType.PULSE:
            if self.phase < np.pi:
                output = 1
            else:
                output = -1

        if self.motor_type == MotorType.SINE:
            output = np.sin(self.phase)

        return output * self.amp 

class Creature:
    def __init__(self, gene_count):
        self.spec = genome.Genome.get_gene_spec()
        self.dna = genome.Genome.get_random_genome(len(self.spec), gene_count)
        self.flat_links = None
        self.exp_links = None
        self.motors = None
        self.start_position = None
        self.last_position = None

        self.max_height = 0 # max height 
        self.baseline_height = None

    def get_flat_links(self):
        if self.flat_links == None:
            gdicts = genome.Genome.get_genome_dicts(self.dna, self.spec)
            self.flat_links = genome.Genome.genome_to_links(gdicts)
        return self.flat_links
    
    def get_expanded_links(self):
        self.get_flat_links()
        if self.exp_links is not None:
            return self.exp_links
        
        exp_links = [self.flat_links[0]]
        genome.Genome.expandLinks(self.flat_links[0], 
                                self.flat_links[0].name, 
                                self.flat_links, 
                                exp_links)
        self.exp_links = exp_links
        return self.exp_links

    def to_xml(self):
        self.get_expanded_links()
        domimpl = getDOMImplementation()
        adom = domimpl.createDocument(None, "start", None)
        robot_tag = adom.createElement("robot")
        for link in self.exp_links:
            robot_tag.appendChild(link.to_link_element(adom))
        first = True
        for link in self.exp_links:
            if first:# skip the root node! 
                first = False
                continue
            robot_tag.appendChild(link.to_joint_element(adom))
        robot_tag.setAttribute("name", "pepe") #  choose a name!
        return '<?xml version="1.0"?>' + robot_tag.toprettyxml()

    def get_motors(self):
        self.get_expanded_links()
        if self.motors == None:
            motors = []
            for i in range(1, len(self.exp_links)):
                l = self.exp_links[i]
                m = Motor(l.control_waveform, l.control_amp,  l.control_freq)
                motors.append(m)
            self.motors = motors 
        return self.motors 
    
    def update_position(self, pos):
        if self.start_position == None:
            self.start_position = pos
        else:
            self.last_position = pos

    def get_distance_travelled(self):
        if self.start_position is None or self.last_position is None:
            return 0
        p1 = np.asarray(self.start_position)
        p2 = np.asarray(self.last_position)
        dist = np.linalg.norm(p1-p2)
        return dist 

    def update_dna(self, dna):
        self.dna = dna
        self.flat_links = None
        self.exp_links = None
        self.motors = None
        self.start_position = None
        self.last_position = None
        self.max_height = 0
        self.baseline_height = None

    # update max height climbed by creatures to determine fitness score. args: (x, y, z) position
    def update_max_height(self, pos):
        if pos is not None and len(pos) >= 3:
            current_height = pos[2]
            
            # Set baseline on first call (settled position after spawn)
            if self.baseline_height is None:
                self.baseline_height = current_height
                self.max_height = 0  # Start from 0 relative to baseline
            else:
                # Track height relative to baseline (actual climbing)
                relative_height = current_height - self.baseline_height
                if relative_height > self.max_height:
                    self.max_height = relative_height

    # get max height reached by creature. returns: max z-coordinate during simulation
    def get_max_height(self):
        """
        Get the maximum height achieved by the creature.
        Used as fitness function for mountain climbing.
        
        Returns:
            float: Maximum z-coordinate reached during simulation
        """
        return self.max_height
    
    def get_distance_to_center(self):
        """
        Calculate distance from mountain center (0, 0).
        Used for navigation component of fitness.

        Returns:
            float: Distance from current position to (0, 0)
        """
        if self.last_position is None:
            return 5.0  # Starting distance from spawn point at (-5, 0, 3)
        x, y = self.last_position[0], self.last_position[1]
        import math
        return math.sqrt(x*x + y*y)
    
    def get_proximity_score(self):
        """
        Calculate proximity score (inverse of distance to center).
        Higher score = closer to mountain peak at (0, 0).

        Returns:
            float: Proximity score (0 to 5.0)
        """
        distance = self.get_distance_to_center()
        return max(0, 5.0 - distance)
    
    def get_hybrid_fitness(self):
        """
        Calculate hybrid fitness combining climbing + navigation.
        Fitness = height_climbed + proximity_to_center
        
        Returns:
            float: Combined fitness score
        """
        # Component 1: Height climbed (vertical progress)
        height_component = self.max_height
        
        # Component 2: Proximity to center (horizontal progress)
        proximity_component = self.get_proximity_score()
        
        # Combined fitness with equal weight (1.0)
        return height_component + (proximity_component * 1.0)