"""
Visualize the diagnostic creature in PyBullet to confirm floating limb issue.
"""

import pybullet as p
import time

def main():
    # Connect to PyBullet
    pid = p.connect(p.DIRECT)
    p.setGravity(0, 0, -10, physicsClientId=pid)

    # Create floor
    floor = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=pid)
    p.createMultiBody(0, floor, physicsClientId=pid)

    # Load the diagnostic creature
    robot = p.loadURDF('diagnostic_creature.urdf', [0, 0, 2], physicsClientId=pid)
    print("URDF loaded successfully!")

    # Get info about each link
    num_joints = p.getNumJoints(robot, physicsClientId=pid)
    print(f"\nNumber of joints: {num_joints}")

    # Get base (link 0) position
    base_pos, base_orn = p.getBasePositionAndOrientation(robot, physicsClientId=pid)
    print(f"\nINITIAL POSITIONS:")
    print(f"  Base link (0) position: {base_pos}")

    for i in range(num_joints):
        link_state = p.getLinkState(robot, i, physicsClientId=pid)
        print(f"  Child link ({i}) world position: {link_state[0]}")

    # Calculate expected vs actual
    print(f"\nANALYSIS:")
    print(f"  Base is at Z={base_pos[2]:.4f}")
    print(f"  Parent cylinder length: 0.65, so extends from Z={base_pos[2]-0.325:.4f} to Z={base_pos[2]+0.325:.4f}")

    if num_joints > 0:
        link_state = p.getLinkState(robot, 0, physicsClientId=pid)
        child_pos = link_state[0]
        print(f"\n  Child link is at: X={child_pos[0]:.4f}, Y={child_pos[1]:.4f}, Z={child_pos[2]:.4f}")

        # Check if child is properly connected
        x_offset = child_pos[0] - base_pos[0]
        z_offset = child_pos[2] - base_pos[2]
        print(f"\n  Offset from base: X={x_offset:.4f}, Z={z_offset:.4f}")

        if abs(x_offset) > 0.1:
            print(f"\n  ⚠️  PROBLEM: Child is offset {x_offset:.4f} units in X-axis!")
            print(f"      This means it's positioned BESIDE the parent, not at its end.")

    # Run simulation and check if parts separate
    print("\n\nSIMULATION TEST (240 steps):")
    for _ in range(240):
        p.stepSimulation(physicsClientId=pid)

    base_pos_after, _ = p.getBasePositionAndOrientation(robot, physicsClientId=pid)
    print(f"  Base after settling: Z={base_pos_after[2]:.4f}")

    if num_joints > 0:
        link_state_after = p.getLinkState(robot, 0, physicsClientId=pid)
        child_pos_after = link_state_after[0]
        print(f"  Child after settling: Z={child_pos_after[2]:.4f}")

        # Check relative positions
        rel_z = child_pos_after[2] - base_pos_after[2]
        rel_x = child_pos_after[0] - base_pos_after[0]
        print(f"\n  Relative offset after settling: X={rel_x:.4f}, Z={rel_z:.4f}")

    p.disconnect(pid)

if __name__ == "__main__":
    main()
