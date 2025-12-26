import pybullet as p
from multiprocessing import Pool
import math

class Simulation: 
    def __init__(self, sim_id=0):
        self.physicsClientId = p.connect(p.DIRECT)
        self.sim_id = sim_id

    def run_creature(self, cr, iterations=2400):
        pid = self.physicsClientId
        p.resetSimulation(physicsClientId=pid)
        p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=pid)

        p.setGravity(0, 0, -10, physicsClientId=pid)

        # Create arena and mountain environment instead of flat plane
        arena_size = 40
        floor_id = make_arena(arena_size=arena_size, wall_height=1, physicsClientId=pid)

        # Set search path for mountain URDF files
        p.setAdditionalSearchPath('shapes/', physicsClientId=pid)

        # Load mountain
        mountain_position = (0, 0, -1)
        mountain_orientation = p.getQuaternionFromEuler((0, 0, 0))
        mountain_id = load_mountain("gaussian_pyramid.urdf", mountain_position, mountain_orientation, physicsClientId=pid)

        xml_file = 'temp' + str(self.sim_id) + '.urdf'
        xml_str = cr.to_xml()
        with open(xml_file, 'w') as f:
            f.write(xml_str)

        try:
            cid = p.loadURDF(xml_file, physicsClientId=pid)
        except Exception as e:
            # Invalid URDF - creature gets 0 fitness
            cr.update_final_position((0, 0, 0))
            return

        # Spawn creature at base of mountain - calculate correct surface height
        # Mountain uses Gaussian: height = 5 * exp(-(x² + y²) / 18) - 1
        spawn_x = 1.5  # Spawn on steeper part of mountain for better climbing gradient
        spawn_y = 0.0
        # Calculate surface height at spawn position
        surface_z = 5.0 * math.exp(-(spawn_x**2 + spawn_y**2) / 18.0) - 1.0
        spawn_z = surface_z + 0.5  # Spawn 0.5 units above surface (prevents crash)
        p.resetBasePositionAndOrientation(cid, [spawn_x, spawn_y, spawn_z], [0, 0, 0, 1], physicsClientId=pid)

        # Store initial distance to peak for progress tracking
        cr.initial_distance_to_peak = math.sqrt(spawn_x**2 + spawn_y**2)  # Distance from spawn to (0, 0)

        # Phase 1: Brief settling period (480 steps = 0.5 seconds)
        for step in range(480):
            p.stepSimulation(physicsClientId=pid)

        # Phase 2: Main simulation with enhanced tracking
        # First call to update_max_height will set baseline to settled position
        # Wrap in try-except to handle physics instability (creature exploding)
        try:
            for step in range(480, iterations):
                p.stepSimulation(physicsClientId=pid)
                if step % 24 == 0:
                    self.update_motors(cid=cid, cr=cr)

                # Get base position for horizontal tracking (x, y)
                base_pos, orn = p.getBasePositionAndOrientation(cid, physicsClientId=pid)
                cr.update_position(base_pos)

                # Get LOWEST point of creature using AABB (prevents tall creatures from cheating)
                # This measures from the bottom of the creature, not the center
                lowest_z = self.get_lowest_point(cid, pid)
                lowest_pos = (base_pos[0], base_pos[1], lowest_z)
                cr.update_max_height(lowest_pos)  # Track height of lowest point

                # Check ground contact with mountain or floor
                is_grounded = self.check_ground_contact(cid, mountain_id, floor_id, pid)
                cr.update_grounded_state(is_grounded)
                cr.update_grounded_height(lowest_pos)  # Only counts when grounded

            # Store final position using lowest point for fitness calculation
            base_pos, _ = p.getBasePositionAndOrientation(cid, physicsClientId=pid)
            lowest_z = self.get_lowest_point(cid, pid)
            final_pos = (base_pos[0], base_pos[1], lowest_z)
            cr.update_final_position(final_pos)
        except Exception as e:
            # Physics became unstable - creature gets 0 fitness
            # This can happen with complex/unstable creature geometries
            cr.update_final_position((spawn_x, spawn_y, spawn_z))  # Use spawn position

    def check_ground_contact(self, creature_id, mountain_id, floor_id, pid):
        """
        Check if the creature is in contact with the mountain or floor.
        Uses PyBullet's getContactPoints to detect collisions.

        This implements the PDF requirement: "without cheating and flying into the air"

        Returns:
            bool: True if creature has contact with ground/mountain
        """
        # Check contact with mountain
        mountain_contacts = p.getContactPoints(creature_id, mountain_id, physicsClientId=pid)
        if mountain_contacts and len(mountain_contacts) > 0:
            return True

        # Check contact with floor
        floor_contacts = p.getContactPoints(creature_id, floor_id, physicsClientId=pid)
        if floor_contacts and len(floor_contacts) > 0:
            return True

        return False

    def get_lowest_point(self, creature_id, pid):
        """
        Get the lowest Z coordinate of the ENTIRE creature by checking all links.
        PyBullet's getAABB with linkIndex=-1 only returns the base link AABB.
        We need to iterate through all links to get the true lowest point.

        This prevents tall creatures from gaming the fitness by measuring
        from their center instead of their actual ground contact point.

        Returns:
            float: Lowest Z coordinate across all parts of the creature
        """
        # Start with base link AABB
        aabb_min, aabb_max = p.getAABB(creature_id, -1, physicsClientId=pid)
        lowest_z = aabb_min[2]

        # Check all child links
        num_joints = p.getNumJoints(creature_id, physicsClientId=pid)
        for link_idx in range(num_joints):
            link_aabb_min, link_aabb_max = p.getAABB(creature_id, link_idx, physicsClientId=pid)
            if link_aabb_min[2] < lowest_z:
                lowest_z = link_aabb_min[2]

        return lowest_z

    def update_motors(self, cid, cr):
        """
        cid is the id in the physics engine
        cr is a creature object
        """
        for jid in range(p.getNumJoints(cid,
                                        physicsClientId=self.physicsClientId)):
            m = cr.get_motors()[jid]

            p.setJointMotorControl2(cid, jid,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=m.get_output(),
                    force=20,  # Increased from 12 for more powerful movement
                    physicsClientId=self.physicsClientId)
        

    # You can add this to the Simulation class:
    def eval_population(self, pop, iterations):
        for cr in pop.creatures:
            self.run_creature(cr, 2400) 


class ThreadedSim():
    def __init__(self, pool_size):
        self.sims = [Simulation(i) for i in range(pool_size)]

    @staticmethod
    def static_run_creature(sim, cr, iterations):
        sim.run_creature(cr, iterations)
        return cr
    
    def eval_population(self, pop, iterations):
        """
        pop is a Population object
        iterations is frames in pybullet to run for at 240fps
        """
        pool_args = [] 
        start_ind = 0
        pool_size = len(self.sims)
        while start_ind < len(pop.creatures):
            this_pool_args = []
            for i in range(start_ind, start_ind + pool_size):
                if i == len(pop.creatures):# the end
                    break
                # work out the sim ind
                sim_ind = i % len(self.sims)
                this_pool_args.append([
                            self.sims[sim_ind], 
                            pop.creatures[i], 
                            iterations]   
                )
            pool_args.append(this_pool_args)
            start_ind = start_ind + pool_size

        new_creatures = []
        for pool_argset in pool_args:
            with Pool(pool_size) as p:
                # it works on a copy of the creatures, so receive them
                creatures = p.starmap(ThreadedSim.static_run_creature, pool_argset)
                # and now put those creatures back into the main 
                # self.creatures array
                new_creatures.extend(creatures)
        pop.creatures = new_creatures



# Create arena with walls, returns: floor body ID
def make_arena(arena_size=10, wall_height=1, physicsClientId=None):
    wall_thickness = 0.5
    
    # Create floor
    floor_collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_BOX, 
        halfExtents=[arena_size/2, arena_size/2, wall_thickness],
        physicsClientId=physicsClientId
    )
    floor_visual_shape = p.createVisualShape(
        shapeType=p.GEOM_BOX, 
        halfExtents=[arena_size/2, arena_size/2, wall_thickness], 
        rgbaColor=[1, 1, 0, 1],
        physicsClientId=physicsClientId
    )
    floor_body = p.createMultiBody(
        baseMass=0, 
        baseCollisionShapeIndex=floor_collision_shape, 
        baseVisualShapeIndex=floor_visual_shape, 
        basePosition=[0, 0, -wall_thickness],
        physicsClientId=physicsClientId
    )

    # create walls 
    wall_collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_BOX, 
        halfExtents=[arena_size/2, wall_thickness/2, wall_height/2],
        physicsClientId=physicsClientId
    )
    wall_visual_shape = p.createVisualShape(
        shapeType=p.GEOM_BOX, 
        halfExtents=[arena_size/2, wall_thickness/2, wall_height/2], 
        rgbaColor=[0.7, 0.7, 0.7, 1],
        physicsClientId=physicsClientId
    )
    
    p.createMultiBody(
        baseMass=0, 
        baseCollisionShapeIndex=wall_collision_shape, 
        baseVisualShapeIndex=wall_visual_shape, 
        basePosition=[0, arena_size/2, wall_height/2],
        physicsClientId=physicsClientId
    )
    p.createMultiBody(
        baseMass=0, 
        baseCollisionShapeIndex=wall_collision_shape, 
        baseVisualShapeIndex=wall_visual_shape, 
        basePosition=[0, -arena_size/2, wall_height/2],
        physicsClientId=physicsClientId
    )

    # left and right walls
    wall_collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_BOX, 
        halfExtents=[wall_thickness/2, arena_size/2, wall_height/2],
        physicsClientId=physicsClientId
    )
    wall_visual_shape = p.createVisualShape(
        shapeType=p.GEOM_BOX, 
        halfExtents=[wall_thickness/2, arena_size/2, wall_height/2], 
        rgbaColor=[0.7, 0.7, 0.7, 1],
        physicsClientId=physicsClientId
    )
    
    p.createMultiBody(
        baseMass=0, 
        baseCollisionShapeIndex=wall_collision_shape, 
        baseVisualShapeIndex=wall_visual_shape, 
        basePosition=[arena_size/2, 0, wall_height/2],
        physicsClientId=physicsClientId
    )
    p.createMultiBody(
        baseMass=0, 
        baseCollisionShapeIndex=wall_collision_shape, 
        baseVisualShapeIndex=wall_visual_shape, 
        basePosition=[-arena_size/2, 0, wall_height/2],
        physicsClientId=physicsClientId
    )
    
    return floor_body

# load mountain URDF, returns: mountain body ID
def load_mountain(mountain_file, position, orientation, physicsClientId=None):
        mountain = p.loadURDF(
            mountain_file, 
            position, 
            orientation, 
            useFixedBase=1,
            physicsClientId=physicsClientId
        )
        return mountain