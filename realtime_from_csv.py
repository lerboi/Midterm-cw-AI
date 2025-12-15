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
    make_arena(arena_size=arena_size, wall_height=1, physicsClientId=pid)
    
    # Set search path for mountain URDF files
    p.setAdditionalSearchPath('shapes/', physicsClientId=pid)
    
    # Load mountain
    mountain_position = (0, 0, -1)
    mountain_orientation = p.getQuaternionFromEuler((0, 0, 0))
    load_mountain("gaussian_pyramid.urdf", mountain_position, mountain_orientation, physicsClientId=pid)

    # generate a random creature
    cr = creature.Creature(gene_count=1)
    dna = genome.Genome.from_csv(csv_file)
    cr.update_dna(dna)
    # save it to XML
    with open('test.urdf', 'w') as f:
        f.write(cr.to_xml())
    # load it into the sim
    rob1 = p.loadURDF('test.urdf', physicsClientId=pid)
    
    # Spawn at base of mountain (not dropped from height)
    p.resetBasePositionAndOrientation(rob1, [-7.8, 0, 3], [0, 0, 0, 1], physicsClientId=pid)
    
    # Brief settling period
    for i in range(480):
        p.stepSimulation(physicsClientId=pid)
    
    start_pos, orn = p.getBasePositionAndOrientation(rob1, physicsClientId=pid)

    # iterate 
    elapsed_time = 0
    wait_time = 1.0/240 # seconds
    total_time = 30 # seconds
    step = 0
    max_height = 0
    baseline_height = None
    
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
                            physicsClientId=pid)
            new_pos, orn = p.getBasePositionAndOrientation(rob1, physicsClientId=pid)
            
            # Track height relative to baseline (same as training)
            if baseline_height is None:
                baseline_height = new_pos[2]
                max_height = 0
            else:
                relative_height = new_pos[2] - baseline_height
                if relative_height > max_height:
                    max_height = relative_height
            
            print(f"Current Z: {new_pos[2]:.3f}, Baseline: {baseline_height:.3f}, Climb: {max_height:.3f}")
        
        time.sleep(wait_time)
        elapsed_time += wait_time
        if elapsed_time > total_time:
            break

    print(f"\nFINAL MAX HEIGHT REACHED: {max_height:.3f}")



if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python realtime_from_csv.py csv_filename"
    main(sys.argv[1])