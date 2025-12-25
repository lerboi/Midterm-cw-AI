import os 
import genome
import sys
import creature
import pybullet as p
import time 
import random
import numpy as np

## ... usual starter code to create a sim and floor
def main(csv_file):
    assert os.path.exists(csv_file), "Tried to load " + csv_file + " but it does not exists"

    pid = p.connect(p.GUI)
    p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=pid)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=pid)
    
    # Create arena and mountain environment
    p.setGravity(0, 0, -10, physicsClientId=pid)
    arena_size = 40
    
    # Import arena and mountain functions from simulation
    from simulation import make_arena, load_mountain
    floor_id = make_arena(arena_size=arena_size, wall_height=1, physicsClientId=pid)

    # Set search path for mountain URDF files
    p.setAdditionalSearchPath('shapes/', physicsClientId=pid)

    # Load mountain
    mountain_position = (0, 0, -1)
    mountain_orientation = p.getQuaternionFromEuler((0, 0, 0))
    mountain_id = load_mountain("gaussian_pyramid.urdf", mountain_position, mountain_orientation, physicsClientId=pid)

    # generate a random creature
    cr = creature.Creature(gene_count=1)
    dna = genome.Genome.from_csv(csv_file)
    cr.update_dna(dna)
    # save it to XML
    with open('test.urdf', 'w') as f:
        f.write(cr.to_xml())
    # load it into the sim
    rob1 = p.loadURDF('test.urdf', physicsClientId=pid)
    
    # Spawn at base of mountain (same as training in simulation.py)
    p.resetBasePositionAndOrientation(rob1, [5, 0, 3], [0, 0, 0, 1], physicsClientId=pid)

    # Store initial distance to peak for progress tracking
    cr.initial_distance_to_peak = 5.0  # Distance from (5, 0) to (0, 0)

    # Brief settling period
    for i in range(480):
        p.stepSimulation(physicsClientId=pid)

    # Helper function for ground contact (same as simulation.py)
    def check_ground_contact(creature_id, mountain_id, floor_id, pid):
        mountain_contacts = p.getContactPoints(creature_id, mountain_id, physicsClientId=pid)
        if mountain_contacts and len(mountain_contacts) > 0:
            return True
        floor_contacts = p.getContactPoints(creature_id, floor_id, physicsClientId=pid)
        if floor_contacts and len(floor_contacts) > 0:
            return True
        return False

    # Helper function to get lowest point of ENTIRE creature (same as simulation.py)
    def get_lowest_point(creature_id, pid):
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

    # iterate
    elapsed_time = 0
    wait_time = 1.0/240 # seconds
    total_time = 30 # seconds
    step = 0

    while True:
        p.stepSimulation(physicsClientId=pid)
        step += 1
        if step % 24 == 0:
            motors = cr.get_motors()
            assert len(motors) == p.getNumJoints(rob1, physicsClientId=pid), "Something went wrong"
            for jid in range(p.getNumJoints(rob1, physicsClientId=pid)):
                mode = p.VELOCITY_CONTROL
                vel = motors[jid].get_output()
                p.setJointMotorControl2(rob1,
                            jid,
                            controlMode=mode,
                            targetVelocity=vel,
                            force=12,
                            physicsClientId=pid)

            # Track using creature's methods (same as training)
            base_pos, orn = p.getBasePositionAndOrientation(rob1, physicsClientId=pid)
            cr.update_position(base_pos)

            # Get LOWEST point of creature using AABB (prevents tall creatures from cheating)
            lowest_z = get_lowest_point(rob1, pid)
            lowest_pos = (base_pos[0], base_pos[1], lowest_z)
            cr.update_max_height(lowest_pos)

            # Check ground contact
            is_grounded = check_ground_contact(rob1, mountain_id, floor_id, pid)
            cr.update_grounded_state(is_grounded)
            cr.update_grounded_height(lowest_pos)

            # Display current stats
            fitness = cr.get_climbing_fitness()
            print(f"Lowest Z: {lowest_z:.2f}, Base Z: {base_pos[2]:.2f}, Grounded: {is_grounded}, Fitness: {fitness:.3f}")

        time.sleep(wait_time)
        elapsed_time += wait_time
        if elapsed_time > total_time:
            break

    # Store final position using lowest point
    base_pos, _ = p.getBasePositionAndOrientation(rob1, physicsClientId=pid)
    lowest_z = get_lowest_point(rob1, pid)
    final_pos = (base_pos[0], base_pos[1], lowest_z)
    cr.update_final_position(final_pos)

    print(f"\nFINAL CLIMBING FITNESS: {cr.get_climbing_fitness():.3f}")



if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python realtime_from_csv.py csv_filename"
    main(sys.argv[1])