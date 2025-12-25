import genome
from xml.dom.minidom import getDOMImplementation
from enum import Enum
import numpy as np
import math

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

        # NEW: Enhanced tracking for improved fitness (PDF Part B requirements)
        self.final_position = None  # Track final position instead of max
        self.grounded_height_sum = 0  # Sum of heights when grounded
        self.grounded_steps = 0  # Number of steps creature was grounded
        self.is_grounded = False  # Current ground contact state
        self.initial_distance_to_peak = 3.0  # Starting distance from (-3, 0) to (0, 0)
        self.final_distance_to_peak = 3.0  # Final distance to peak

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
        # Reset new tracking variables
        self.final_position = None
        self.grounded_height_sum = 0
        self.grounded_steps = 0
        self.is_grounded = False
        self.initial_distance_to_peak = 3.0
        self.final_distance_to_peak = 3.0

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
            return 3.0  # Starting distance from spawn point at (-3, 0, 2)
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
        return max(0, 3.0 - distance)
    
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

    # =========================================================================
    # NEW FITNESS METHODS - Addressing PDF Part B requirements:
    # "get as high as possible up the mountain, without cheating and flying"
    # =========================================================================

    @staticmethod
    def get_mountain_height(x, y):
        """
        Calculate the mountain surface height at position (x, y).
        Uses the same Gaussian formula as prepare_shapes.py.
        Mountain is positioned at z=-1, with sigma=3, height=5.

        Returns:
            float: Height of mountain surface at (x, y)
        """
        sigma = 3.0
        height = 5.0
        mountain_base = -1.0
        # Gaussian: height * exp(-(x² + y²) / (2 * sigma²))
        surface_height = height * math.exp(-((x**2 + y**2) / (2 * sigma**2)))
        return surface_height + mountain_base

    def update_grounded_state(self, is_grounded):
        """
        Update whether creature is in contact with ground/mountain.
        Called by simulation when ground contact is detected.

        Args:
            is_grounded: Boolean indicating ground contact
        """
        self.is_grounded = is_grounded

    def update_final_position(self, pos):
        """
        Update final position (called at end of simulation).
        This is used for FINAL position fitness instead of MAX position.

        Args:
            pos: (x, y, z) tuple of final position
        """
        if pos is not None and len(pos) >= 3:
            self.final_position = pos
            self.final_distance_to_peak = math.sqrt(pos[0]**2 + pos[1]**2)

    def update_grounded_height(self, pos):
        """
        Track height only when creature is grounded.
        Prevents "flying" or "jumping" from being rewarded.

        Args:
            pos: (x, y, z) position tuple
        """
        if pos is not None and len(pos) >= 3 and self.is_grounded:
            self.grounded_height_sum += pos[2]
            self.grounded_steps += 1

    def get_final_height(self):
        """
        Get the FINAL height of the creature (not max).
        Rewards sustainable climbing, not brief spikes.

        Returns:
            float: Final z-coordinate relative to baseline
        """
        if self.final_position is None or self.baseline_height is None:
            return 0
        return self.final_position[2] - self.baseline_height

    def get_height_above_surface(self):
        """
        Calculate creature's height above the mountain surface.
        Prevents tall stationary creatures from scoring high.

        Returns:
            float: Height above the mountain surface at current position
        """
        if self.final_position is None:
            return 0
        x, y, z = self.final_position
        surface_z = self.get_mountain_height(x, y)
        return z - surface_z

    def get_average_grounded_height(self):
        """
        Get average height while grounded.
        Only counts time when creature was touching the surface.

        Returns:
            float: Average height while in contact with ground
        """
        if self.grounded_steps == 0:
            return 0
        return (self.grounded_height_sum / self.grounded_steps) - (self.baseline_height or 0)

    def get_progress_toward_peak(self):
        """
        Calculate how much closer the creature moved toward the peak.
        Rewards actual locomotion toward (0, 0).

        Returns:
            float: Progress score (positive = moved toward peak)
        """
        # Progress = initial_distance - final_distance
        # Positive means creature moved closer to peak
        return self.initial_distance_to_peak - self.final_distance_to_peak

    def get_climbing_fitness(self):
        """
        NEW FITNESS FUNCTION - Designed to reward actual climbing behavior.

        Addresses PDF Part B requirement: "get as high as possible up the
        mountain, without cheating and flying into the air"

        Components:
        1. Final height (not max) - rewards sustainable position
        2. Height above surface - ensures creature is ON the mountain
        3. Progress toward peak - rewards moving toward (0,0)
        4. Grounding bonus - rewards staying in contact with surface

        Returns:
            float: Combined climbing fitness score
        """
        # Component 1: FINAL height relative to baseline (not max)
        # This prevents brief spikes from jumping/falling from being rewarded
        final_height = self.get_final_height()

        # Component 2: Height above mountain surface
        # Small values = creature is ON the surface (good for climbing)
        # Large values = creature is floating/flying (bad)
        height_above_surface = self.get_height_above_surface()

        # Penalize if creature is too high above surface (likely not climbing)
        # Ideal: creature is within 0.5 units of surface
        surface_penalty = max(0, height_above_surface - 0.5) * 0.5

        # Component 3: Progress toward peak
        # Rewards creatures that moved closer to (0, 0)
        progress = self.get_progress_toward_peak()

        # Component 4: Grounding bonus
        # Rewards creatures that stayed in contact with surface
        grounding_ratio = self.grounded_steps / max(1, 1920)  # ~1920 tracking steps
        grounding_bonus = grounding_ratio * 0.5

        # Combined fitness:
        # - Final height (main reward for climbing)
        # - Progress toward peak (reward for moving to center)
        # - Grounding bonus (reward for staying on surface)
        # - Surface penalty (penalize floating/flying)
        fitness = final_height + (progress * 1.0) + grounding_bonus - surface_penalty

        # Ensure non-negative fitness
        return max(0, fitness)