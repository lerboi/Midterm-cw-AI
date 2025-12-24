"""
Diagnostic script to inspect URDF generation and identify floating limb issues.
Generates a simple 2-link creature and outputs the URDF for analysis.
"""

import genome
import creature
import numpy as np

def create_simple_creature():
    """Create a simple 2-gene creature with known values for analysis."""

    # Get the gene spec to understand the structure
    spec = genome.Genome.get_gene_spec()
    print("=" * 60)
    print("GENE SPEC ANALYSIS")
    print("=" * 60)
    for key, val in spec.items():
        print(f"  {key}: scale={val['scale']}, index={val['ind']}")

    # Create a creature with 2 genes (2 links)
    cr = creature.Creature(gene_count=2)

    # Set known DNA values for predictable output
    # Gene structure: 17 values per gene (0-1 range)
    gene1 = np.array([
        0.5,  # link-shape
        0.5,  # link-length -> 0.5 * 1.3 = 0.65
        0.5,  # link-radius -> 0.5 * 0.65 = 0.325
        0.0,  # link-recurrence -> 0 * 2 = 0 -> 1 copy
        0.5,  # link-mass
        0.5,  # joint-type
        0.0,  # joint-parent -> links to first available parent
        0.5,  # joint-axis-xyz
        0.0,  # joint-origin-rpy-1
        0.0,  # joint-origin-rpy-2
        0.0,  # joint-origin-rpy-3
        0.5,  # joint-origin-xyz-1
        0.0,  # joint-origin-xyz-2
        0.0,  # joint-origin-xyz-3
        0.5,  # control-waveform
        0.5,  # control-amp
        0.5,  # control-freq
    ])

    gene2 = np.array([
        0.5,  # link-shape
        0.8,  # link-length -> 0.8 * 1.3 = 1.04
        0.5,  # link-radius -> 0.5 * 0.65 = 0.325
        0.0,  # link-recurrence -> 1 copy
        0.5,  # link-mass
        0.5,  # joint-type
        0.0,  # joint-parent
        0.5,  # joint-axis-xyz
        0.0,  # joint-origin-rpy-1
        0.0,  # joint-origin-rpy-2
        0.0,  # joint-origin-rpy-3
        0.5,  # joint-origin-xyz-1
        0.0,  # joint-origin-xyz-2
        0.0,  # joint-origin-xyz-3
        0.5,  # control-waveform
        0.5,  # control-amp
        0.5,  # control-freq
    ])

    cr.dna = [gene1, gene2]

    return cr

def analyze_links(cr):
    """Analyze the flat and expanded links."""
    print("\n" + "=" * 60)
    print("FLAT LINKS (before expansion)")
    print("=" * 60)

    flat_links = cr.get_flat_links()
    for i, link in enumerate(flat_links):
        print(f"\nLink {i}: '{link.name}'")
        print(f"  parent_name: '{link.parent_name}'")
        print(f"  recurrence: {link.recur}")
        print(f"  link_length: {link.link_length:.4f}")
        print(f"  link_radius: {link.link_radius:.4f}")
        print(f"  joint_origin_xyz: ({link.joint_origin_xyz_1:.4f}, {link.joint_origin_xyz_2:.4f}, {link.joint_origin_xyz_3:.4f})")
        print(f"  joint_origin_rpy: ({link.joint_origin_rpy_1:.4f}, {link.joint_origin_rpy_2:.4f}, {link.joint_origin_rpy_3:.4f})")

    print("\n" + "=" * 60)
    print("EXPANDED LINKS (after recurrence)")
    print("=" * 60)

    exp_links = cr.get_expanded_links()
    for i, link in enumerate(exp_links):
        print(f"\nLink {i}: '{link.name}'")
        print(f"  parent_name: '{link.parent_name}'")
        print(f"  sibling_ind: {link.sibling_ind}")
        print(f"  link_length: {link.link_length:.4f}")
        print(f"  link_radius: {link.link_radius:.4f}")

def analyze_urdf(cr):
    """Generate and analyze the URDF output."""
    print("\n" + "=" * 60)
    print("GENERATED URDF")
    print("=" * 60)

    xml_str = cr.to_xml()
    print(xml_str)

    # Save to file for inspection
    with open('diagnostic_creature.urdf', 'w') as f:
        f.write(xml_str)
    print("\nURDF saved to: diagnostic_creature.urdf")

def analyze_joint_calculation():
    """Manually trace through joint origin calculation."""
    print("\n" + "=" * 60)
    print("JOINT ORIGIN CALCULATION ANALYSIS")
    print("=" * 60)

    # Simulate what happens in to_joint_element()
    # Using example values

    parent_link_length = 0.65  # Parent's actual length
    child_link_length = 1.04   # Child's link_length (what code currently uses)
    joint_origin_xyz_1 = 0.5   # Gene value (0-1) * scale (1) = 0.5

    # Current calculation (from genome.py:344-347)
    xyz_1_current = child_link_length * 0.5 + joint_origin_xyz_1 * 0.1
    xyz_2_current = 0.0 * 0.1  # Assuming 0
    xyz_3_current = 0.0 * 0.1  # Assuming 0

    print(f"\nExample scenario:")
    print(f"  Parent link length: {parent_link_length}")
    print(f"  Child link length: {child_link_length}")
    print(f"  joint_origin_xyz_1 gene value: {joint_origin_xyz_1}")

    print(f"\nCurrent calculation (using CHILD's length):")
    print(f"  xyz_1 = child_length * 0.5 + gene * 0.1")
    print(f"  xyz_1 = {child_link_length} * 0.5 + {joint_origin_xyz_1} * 0.1")
    print(f"  xyz_1 = {xyz_1_current:.4f}")
    print(f"  Joint placed at: ({xyz_1_current:.4f}, {xyz_2_current:.4f}, {xyz_3_current:.4f})")

    print(f"\nProblem analysis:")
    print(f"  - URDF cylinders extend along Z-axis by default")
    print(f"  - Parent cylinder extends from z=-{parent_link_length/2:.4f} to z=+{parent_link_length/2:.4f}")
    print(f"  - To attach at parent's END, joint should be at z=+{parent_link_length/2:.4f}")
    print(f"  - But current code offsets in X-axis (xyz_1), not Z-axis (xyz_3)")
    print(f"  - And uses child's length ({child_link_length}) instead of parent's ({parent_link_length})")

    print(f"\nExpected joint position for proper attachment:")
    print(f"  xyz = (small_offset, small_offset, {parent_link_length/2:.4f})")
    print(f"\nActual joint position from current code:")
    print(f"  xyz = ({xyz_1_current:.4f}, 0, 0)")

    gap = abs(xyz_1_current - parent_link_length/2)
    print(f"\nThis places joint {xyz_1_current:.4f} units along X-axis")
    print(f"When it should be {parent_link_length/2:.4f} units along Z-axis")
    print(f"Result: Child link appears BESIDE parent, not attached to its end")

def main():
    print("URDF DIAGNOSTIC TOOL")
    print("Analyzing creature generation to identify floating limb issues\n")

    # Create simple creature
    cr = create_simple_creature()

    # Analyze links
    analyze_links(cr)

    # Analyze joint calculation
    analyze_joint_calculation()

    # Generate and show URDF
    analyze_urdf(cr)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
The diagnostic reveals potential issues:

1. AXIS MISMATCH:
   - Cylinders extend along Z-axis
   - Joint offset is primarily in X-axis (xyz_1)
   - This places child links BESIDE parents, not at their ends

2. WRONG LENGTH USED:
   - Code uses self.link_length (child's length)
   - Should use parent's link_length to position at parent's endpoint

3. NO PARENT REFERENCE:
   - to_joint_element() only has access to child link data
   - parent_name is just a string, not a reference to parent object
   - Cannot look up parent's link_length

Check the generated URDF file to verify these findings visually.
""")

if __name__ == "__main__":
    main()
