import numpy as np
import copy
import random

class Genome():
    @staticmethod
    def get_random_gene(length):
        gene = np.array([np.random.random() for i in range(length)])
        return gene

    @staticmethod
    def get_random_genome(gene_length, gene_count):
        genome = [Genome.get_random_gene(gene_length) for i in range(gene_count)]
        return genome

    @staticmethod
    def get_fixed_walker_genome(gene_length):
        """
        Create a fixed walker body structure with random motor control genes.

        This implements the PDF suggestion: "start with a fixed design robot,
        and just evolve the motor control parameters"

        Body plan:
        - Gene 0: Body (SPHERE as stable central mass)
        - Gene 1: Legs (CYLINDER legs via recurrence, angled DOWNWARD)

        The walker seed provides SHAPE VARIETY from the start:
        - Spherical body for stability and rolling
        - Cylindrical legs for locomotion

        Only control genes (indices 14-16) are randomized.
        Morphology genes (indices 0-13) are fixed for a functional walker.

        NOTE: Gene scales are:
        - link-length: scale 0.8 (max 0.8 units)
        - link-radius: scale 0.2 (max 0.2 units)
        """
        genome = []

        # Gene 0: Body (root link) - SPHERE for stable central mass
        body_gene = np.zeros(gene_length)
        body_gene[0] = 0.5   # link-shape: 0.5 = SPHERE (0.33-0.66 range)
        body_gene[1] = 0.5   # link-length: 0.5 * 0.8 = 0.4 (affects sphere radius)
        body_gene[2] = 0.8   # link-radius: 0.8 * 0.2 = 0.16 (sphere radius ~0.28)
        body_gene[3] = 0.0   # link-recurrence: 0 (no copies of body)
        body_gene[4] = 0.5   # link-mass: moderate
        body_gene[5] = 0.5   # joint-type: doesn't matter for root
        body_gene[6] = 0.0   # joint-parent: root has no parent
        body_gene[7] = 0.5   # joint-axis-xyz
        body_gene[8] = 0.0   # joint-origin-rpy-1
        body_gene[9] = 0.0   # joint-origin-rpy-2
        body_gene[10] = 0.0  # joint-origin-rpy-3
        body_gene[11] = 0.0  # joint-origin-xyz-1
        body_gene[12] = 0.0  # joint-origin-xyz-2
        body_gene[13] = 0.0  # joint-origin-xyz-3
        # Control genes - randomized for evolution
        body_gene[14] = np.random.random()  # control-waveform
        body_gene[15] = np.random.random()  # control-amp
        body_gene[16] = np.random.random()  # control-freq
        genome.append(body_gene)

        # Gene 1: Legs (attached to body, pointing DOWNWARD) - CYLINDER
        leg_gene = np.zeros(gene_length)
        leg_gene[0] = 0.2    # link-shape: 0.2 = CYLINDER (0-0.33 range)
        leg_gene[1] = 0.75   # link-length: 0.75 * 0.8 = 0.6 (reasonable leg length)
        leg_gene[2] = 0.3    # link-radius: 0.3 * 0.2 = 0.06 (thin legs)
        leg_gene[3] = 1.5    # link-recurrence: 1.5 * 2 = 3, int(3)+1 = 4 legs
        leg_gene[4] = 0.2    # link-mass: light legs
        leg_gene[5] = 0.5    # joint-type: revolute
        leg_gene[6] = 0.0    # joint-parent: attached to body (gene 0)
        leg_gene[7] = 0.2    # joint-axis-xyz: X-axis rotation (legs swing forward/back)
        # KEY FIX: Angle legs DOWNWARD toward ground
        # rpy-1 is multiplied by sibling_ind, spreading legs around body
        leg_gene[8] = 0.125  # joint-origin-rpy-1: π/4 base (spreads to π/2, 3π/4, π per leg)
        leg_gene[9] = 0.375  # joint-origin-rpy-2: 0.375 * 2π = 3π/4 ≈ 135° - angles legs DOWN
        leg_gene[10] = 0.0   # joint-origin-rpy-3: no roll
        # Position legs at sides of body
        leg_gene[11] = 0.8   # joint-origin-xyz-1: X offset (spread outward)
        leg_gene[12] = 0.8   # joint-origin-xyz-2: Y offset (spread outward)
        leg_gene[13] = 0.0   # joint-origin-xyz-3: at body center height
        # Control genes - randomized for evolution
        leg_gene[14] = np.random.random()  # control-waveform
        leg_gene[15] = np.random.random()  # control-amp
        leg_gene[16] = np.random.random()  # control-freq
        genome.append(leg_gene)

        return genome

    @staticmethod
    def get_gene_spec():
        gene_spec =  {"link-shape":{"scale":1},  # 0-0.33=cylinder, 0.33-0.66=sphere, 0.66-1=box
            "link-length": {"scale":0.8},  # Max 0.8 units - reasonable limb size
            "link-radius": {"scale":0.2},  # Max 0.2 units - prevents huge bulky limbs
            "link-recurrence": {"scale":2},  # Max 2 recurrences (3 copies)
            "link-mass": {"scale":1},
            "joint-type": {"scale":1},
            "joint-parent":{"scale":1},
            "joint-axis-xyz": {"scale":1},
            "joint-origin-rpy-1":{"scale":np.pi * 2},
            "joint-origin-rpy-2":{"scale":np.pi * 2},
            "joint-origin-rpy-3":{"scale":np.pi * 2},
            "joint-origin-xyz-1":{"scale":1},
            "joint-origin-xyz-2":{"scale":1},
            "joint-origin-xyz-3":{"scale":1},
            "control-waveform":{"scale":1},
            "control-amp":{"scale":0.25},
            "control-freq":{"scale":3}
            }
        ind = 0
        for key in gene_spec.keys():
            gene_spec[key]["ind"] = ind
            ind = ind + 1
        return gene_spec
    
    @staticmethod
    def get_gene_dict(gene, spec):
        gdict = {}
        for key in spec:
            ind = spec[key]["ind"]
            scale = spec[key]["scale"]
            gdict[key] = gene[ind] * scale
        return gdict

    @staticmethod
    def get_genome_dicts(genome, spec):
        gdicts = []
        for gene in genome:
            gdicts.append(Genome.get_gene_dict(gene, spec))
        return gdicts

    @staticmethod
    def expandLinks(parent_link, uniq_parent_name, flat_links, exp_links):
        children = [l for l in flat_links if l.parent_name == parent_link.name]
        sibling_ind = 1
        for c in children:
            for r in range(int(c.recur)):
                sibling_ind  = sibling_ind +1
                c_copy = copy.copy(c)
                c_copy.parent_name = uniq_parent_name
                uniq_name = c_copy.name + str(len(exp_links))
                #print("exp: ", c.name, " -> ", uniq_name)
                c_copy.name = uniq_name
                c_copy.sibling_ind = sibling_ind
                # Store parent's link_length for proper joint positioning
                c_copy.parent_link_length = parent_link.link_length
                exp_links.append(c_copy)
                assert c.parent_name != c.name, "Genome::expandLinks: link joined to itself: " + c.name + " joins " + c.parent_name
                Genome.expandLinks(c, uniq_name, flat_links, exp_links)

    @staticmethod
    def genome_to_links(gdicts):
        links = []
        link_ind = 0
        parent_names = [str(link_ind)]
        for gdict in gdicts:
            link_name = str(link_ind)
            parent_ind = gdict["joint-parent"] * len(parent_names)
            assert parent_ind < len(parent_names), "genome.py: parent ind too high: " + str(parent_ind) + "got: " + str(parent_names)
            parent_name = parent_names[int(parent_ind)]
            #print("available parents: ", parent_names, "chose", parent_name)
            recur = int(gdict["link-recurrence"]) + 1
            link = URDFLink(name=link_name,
                            parent_name=parent_name,
                            recur=recur,
                            link_shape=gdict["link-shape"],
                            link_length=gdict["link-length"],
                            link_radius=gdict["link-radius"],
                            link_mass=gdict["link-mass"],
                            joint_type=gdict["joint-type"],
                            joint_parent=gdict["joint-parent"],
                            joint_axis_xyz=gdict["joint-axis-xyz"],
                            joint_origin_rpy_1=gdict["joint-origin-rpy-1"],
                            joint_origin_rpy_2=gdict["joint-origin-rpy-2"],
                            joint_origin_rpy_3=gdict["joint-origin-rpy-3"],
                            joint_origin_xyz_1=gdict["joint-origin-xyz-1"],
                            joint_origin_xyz_2=gdict["joint-origin-xyz-2"],
                            joint_origin_xyz_3=gdict["joint-origin-xyz-3"],
                            control_waveform=gdict["control-waveform"],
                            control_amp=gdict["control-amp"],
                            control_freq=gdict["control-freq"])
            links.append(link)
            if link_ind != 0:# don't re-add the first link
                parent_names.append(link_name)
            link_ind = link_ind + 1

        # now just fix the first link so it links to nothing
        links[0].parent_name = "None"
        return links

    @staticmethod
    def crossover(g1, g2, min_genes=2):
        """
        Crossover two genomes to create offspring.

        Args:
            g1: First parent genome
            g2: Second parent genome
            min_genes: Minimum genes in offspring (default 2 for body+legs)
        """
        x1 = random.randint(0, len(g1)-1)
        x2 = random.randint(0, len(g2)-1)
        # Combine genes from both parents as a list (not numpy array)
        # Each gene is a numpy array, genome is a list of genes
        g3 = list(g1[x1:]) + list(g2[x2:])

        # Ensure minimum gene count (need body + legs)
        max_len = max(len(g1), len(g2), min_genes)
        if len(g3) > max_len:
            g3 = g3[0:max_len]

        # If somehow too short, pad with genes from g1
        while len(g3) < min_genes and len(g1) >= min_genes:
            g3.append(g1[len(g3) % len(g1)].copy())

        return g3

    @staticmethod
    def point_mutate(genome, rate, amount):
        # Deep copy to prevent corruption of parent DNA
        new_genome = [gene.copy() for gene in genome]
        for gene in new_genome:
            for i in range(len(gene)):
                if random.random() < rate:
                    # Bidirectional mutation using the amount parameter
                    gene[i] += random.uniform(-amount, amount)
                # Clamp to valid range [0, 1)
                if gene[i] >= 1.0:
                    gene[i] = 0.9999
                if gene[i] < 0.0:
                    gene[i] = 0.0
        return new_genome

    @staticmethod
    def point_mutate_control_only(genome, rate, amount):
        """
        Mutate ONLY the motor control genes (indices 14, 15, 16).
        Preserves body morphology while evolving motor patterns.

        This implements the PDF suggestion to evolve only motor control
        while keeping body structure fixed.

        Control genes:
        - Index 14: control-waveform (pulse vs sine)
        - Index 15: control-amp (motor amplitude)
        - Index 16: control-freq (motor frequency)
        """
        # Control gene indices
        CONTROL_INDICES = [14, 15, 16]

        # Deep copy to prevent corruption of parent DNA
        new_genome = [gene.copy() for gene in genome]
        for gene in new_genome:
            for i in CONTROL_INDICES:
                if i < len(gene) and random.random() < rate:
                    # Bidirectional mutation using the amount parameter
                    gene[i] += random.uniform(-amount, amount)
                    # Clamp to valid range [0, 1)
                    if gene[i] >= 1.0:
                        gene[i] = 0.9999
                    if gene[i] < 0.0:
                        gene[i] = 0.0
        return new_genome

    @staticmethod
    def crossover_control_only(g1, g2):
        """
        Crossover that only exchanges control genes, preserving body structure.
        Takes body from g1 and randomly mixes control genes from both parents.
        """
        # Deep copy g1's structure (preserve morphology)
        g3 = [gene.copy() for gene in g1]

        # Control gene indices
        CONTROL_INDICES = [14, 15, 16]

        # For each gene, randomly take control values from either parent
        for gene_idx in range(min(len(g1), len(g2))):
            for ctrl_idx in CONTROL_INDICES:
                if ctrl_idx < len(g3[gene_idx]):
                    # 50% chance to take from parent 2
                    if random.random() < 0.5:
                        g3[gene_idx][ctrl_idx] = g2[gene_idx][ctrl_idx]

        return g3

    @staticmethod
    def shrink_mutate(genome, rate, min_genes=2):
        """
        Randomly remove a gene from the genome.

        Args:
            genome: List of genes
            rate: Probability of shrinking
            min_genes: Minimum number of genes to maintain (default 2 for body+legs)
        """
        if len(genome) <= min_genes:
            # Don't shrink below minimum - need at least body + legs
            return [gene.copy() for gene in genome]
        if random.random() < rate:
            ind = random.randint(0, len(genome)-1)
            # Create new list without the removed gene
            new_genome = [gene.copy() for i, gene in enumerate(genome) if i != ind]
            return new_genome
        else:
            # Deep copy to prevent corruption of parent DNA
            return [gene.copy() for gene in genome]

    @staticmethod
    def grow_mutate(genome, rate):
        if random.random() < rate:
            gene = Genome.get_random_gene(len(genome[0]))
            # Deep copy existing genes and append new one
            new_genome = [g.copy() for g in genome]
            new_genome.append(gene)
            return new_genome
        else:
            # Deep copy to prevent corruption of parent DNA
            return [gene.copy() for gene in genome]

    @staticmethod
    def to_csv(dna, csv_file):
        csv_str = ""
        for gene in dna:
            for val in gene:
                csv_str = csv_str + str(val) + ","
            csv_str = csv_str + '\n'

        with open(csv_file, 'w') as f:
            f.write(csv_str)

    @staticmethod
    def from_csv(filename):
        csv_str = ''
        with open(filename) as f:
            csv_str = f.read()
        dna = []
        lines = csv_str.split('\n')
        for line in lines:
            vals = line.split(',')
            gene_values = [float(v) for v in vals if v != '']
            if len(gene_values) > 0:
                # Convert to numpy array to match get_random_genome() return type
                gene = np.array(gene_values)
                dna.append(gene)
        return dna

class URDFLink:
    def __init__(self, name, parent_name, recur,
                link_shape=0.1,
                link_length=0.1,
                link_radius=0.1,
                link_mass=0.1,
                joint_type=0.1,
                joint_parent=0.1,
                joint_axis_xyz=0.1,
                joint_origin_rpy_1=0.1,
                joint_origin_rpy_2=0.1,
                joint_origin_rpy_3=0.1,
                joint_origin_xyz_1=0.1,
                joint_origin_xyz_2=0.1,
                joint_origin_xyz_3=0.1,
                control_waveform=0.1,
                control_amp=0.1,
                control_freq=0.1):
        self.name = name
        self.parent_name = parent_name
        self.recur = recur
        self.link_shape = link_shape
        self.link_length=link_length
        self.link_radius=link_radius
        self.link_mass=link_mass
        self.joint_type=joint_type
        self.joint_parent=joint_parent
        self.joint_axis_xyz=joint_axis_xyz
        self.joint_origin_rpy_1=joint_origin_rpy_1
        self.joint_origin_rpy_2=joint_origin_rpy_2
        self.joint_origin_rpy_3=joint_origin_rpy_3
        self.joint_origin_xyz_1=joint_origin_xyz_1
        self.joint_origin_xyz_2=joint_origin_xyz_2
        self.joint_origin_xyz_3=joint_origin_xyz_3
        self.control_waveform=control_waveform
        self.control_amp=control_amp
        self.control_freq=control_freq
        self.sibling_ind = 1
        # Parent's link length - needed for proper joint positioning
        # Will be set during expandLinks() for child links
        self.parent_link_length = 0.1

    def to_link_element(self, adom):
        #         <link name="base_link">
        #     <visual>
        #       <geometry>
        #         <cylinder length="0.6" radius="0.25"/>
        #       </geometry>
        #     </visual>
        #     <collision>
        #       <geometry>
        #         <cylinder length="0.6" radius="0.25"/>
        #       </geometry>
        #     </collision>
        #     <inertial>
        #       <mass value="0.25"/>
        #       <inertia ixx="0.0003" iyy="0.0003" izz="0.0003" ixy="0" ixz="0" iyz="0"/>
        #     </inertial>
        #   </link>

        # Enforce minimum link sizes to prevent physics instability
        min_length = 0.1
        min_radius = 0.05
        actual_length = max(self.link_length, min_length)
        actual_radius = max(self.link_radius, min_radius)

        link_tag = adom.createElement("link")
        link_tag.setAttribute("name", self.name)
        vis_tag = adom.createElement("visual")
        geom_tag = adom.createElement("geometry")

        # SHAPE VARIATION: Use link_shape gene to determine shape type
        # This allows evolution to discover optimal shapes for climbing
        # 0.0-0.33 = cylinder, 0.33-0.66 = sphere, 0.66-1.0 = box
        if self.link_shape <= 0.33:
            # CYLINDER - good for legs, elongated limbs
            shape_tag = adom.createElement("cylinder")
            shape_tag.setAttribute("length", str(actual_length))
            shape_tag.setAttribute("radius", str(actual_radius))
            # Mass = density * volume = pi * r^2 * height
            mass = np.pi * (actual_radius * actual_radius) * actual_length
            # Cylinder inertia
            ixx = (1.0/12.0) * mass * (3 * actual_radius * actual_radius + actual_length * actual_length)
            iyy = ixx
            izz = (1.0/2.0) * mass * actual_radius * actual_radius
        elif self.link_shape <= 0.66:
            # SPHERE - good for body, joints, rounded parts
            shape_tag = adom.createElement("sphere")
            # Sphere radius based on average of length and radius genes
            sphere_radius = (actual_length + actual_radius) / 2
            sphere_radius = max(sphere_radius, min_radius)
            shape_tag.setAttribute("radius", str(sphere_radius))
            # Mass = (4/3) * pi * r^3
            mass = (4.0/3.0) * np.pi * (sphere_radius ** 3)
            # Sphere inertia: I = (2/5) * m * r^2
            ixx = (2.0/5.0) * mass * sphere_radius * sphere_radius
            iyy = ixx
            izz = ixx
        else:
            # BOX - good for flat feet, paddles, stable bases
            shape_tag = adom.createElement("box")
            # Box dimensions: length x width x height
            box_x = actual_length
            box_y = actual_radius * 2  # Width based on radius
            box_z = actual_radius      # Height based on radius
            shape_tag.setAttribute("size", f"{box_x} {box_y} {box_z}")
            # Mass = density * volume = x * y * z
            mass = box_x * box_y * box_z
            # Box inertia
            ixx = (1.0/12.0) * mass * (box_y * box_y + box_z * box_z)
            iyy = (1.0/12.0) * mass * (box_x * box_x + box_z * box_z)
            izz = (1.0/12.0) * mass * (box_x * box_x + box_y * box_y)

        geom_tag.appendChild(shape_tag)
        vis_tag.appendChild(geom_tag)

        coll_tag = adom.createElement("collision")
        c_geom_tag = adom.createElement("geometry")

        # Collision shape (same as visual)
        if self.link_shape <= 0.33:
            c_shape_tag = adom.createElement("cylinder")
            c_shape_tag.setAttribute("length", str(actual_length))
            c_shape_tag.setAttribute("radius", str(actual_radius))
        elif self.link_shape <= 0.66:
            c_shape_tag = adom.createElement("sphere")
            sphere_radius = (actual_length + actual_radius) / 2
            sphere_radius = max(sphere_radius, min_radius)
            c_shape_tag.setAttribute("radius", str(sphere_radius))
        else:
            c_shape_tag = adom.createElement("box")
            box_x = actual_length
            box_y = actual_radius * 2
            box_z = actual_radius
            c_shape_tag.setAttribute("size", f"{box_x} {box_y} {box_z}")

        c_geom_tag.appendChild(c_shape_tag)
        coll_tag.appendChild(c_geom_tag)

        #     <inertial>
        #       <mass value="0.25"/>
        #       <inertia ixx="0.0003" iyy="0.0003" izz="0.0003" ixy="0" ixz="0" iyz="0"/>
        #     </inertial>
        inertial_tag = adom.createElement("inertial")
        mass_tag = adom.createElement("mass")
        mass_tag.setAttribute("value", str(mass))
        inertia_tag = adom.createElement("inertia")
        # Inertia values computed above based on shape type
        inertia_tag.setAttribute("ixx", str(ixx))
        inertia_tag.setAttribute("iyy", str(iyy))
        inertia_tag.setAttribute("izz", str(izz))
        inertia_tag.setAttribute("ixy", "0")
        inertia_tag.setAttribute("ixz", "0")
        inertia_tag.setAttribute("iyz", "0")
        inertial_tag.appendChild(mass_tag)
        inertial_tag.appendChild(inertia_tag)


        link_tag.appendChild(vis_tag)
        link_tag.appendChild(coll_tag)
        link_tag.appendChild(inertial_tag)

        return link_tag

    def to_joint_element(self, adom):
        #           <joint name="base_to_sub2" type="revolute">
        #     <parent link="base_link"/>
        #     <child link="sub_link2"/>
        #     <axis xyz="1 0 0"/>
        #     <limit effort="10" upper="0" lower="10" velocity="1"/>
        #     <origin rpy="0 0 0" xyz="0 0.5 0"/>
        #   </joint>
        joint_tag = adom.createElement("joint")
        joint_tag.setAttribute("name", self.name + "_to_" + self.parent_name)
        # All joints are revolute type for consistent behavior
        joint_tag.setAttribute("type", "revolute")
        parent_tag = adom.createElement("parent")
        parent_tag.setAttribute("link", self.parent_name)
        child_tag = adom.createElement("child")
        child_tag.setAttribute("link", self.name)
        axis_tag = adom.createElement("axis")
        if self.joint_axis_xyz <= 0.33:
            axis_tag.setAttribute("xyz", "1 0 0")
        if self.joint_axis_xyz > 0.33 and self.joint_axis_xyz <= 0.66:
            axis_tag.setAttribute("xyz", "0 1 0")
        if self.joint_axis_xyz > 0.66:
            axis_tag.setAttribute("xyz", "0 0 1")
        
        limit_tag = adom.createElement("limit")
        # effort upper lower velocity
        limit_tag.setAttribute("effort", "10")
        limit_tag.setAttribute("upper", "3.1415")
        limit_tag.setAttribute("lower", "-3.1415")
        limit_tag.setAttribute("velocity", "5")
        # <origin rpy="0 0 0" xyz="0 0.5 0"/>
        orig_tag = adom.createElement("origin")
        
        rpy1 = self.joint_origin_rpy_1 * self.sibling_ind
        rpy = str(rpy1) + " " + str(self.joint_origin_rpy_2) + " " + str(self.joint_origin_rpy_3)
        
        orig_tag.setAttribute("rpy", rpy)
        # Position joint at end of parent link
        # URDF cylinders extend along Z-axis, centered at origin
        # Joint should be at parent's tip with minimal offset
        parent_len = max(self.parent_link_length, 0.1)
        # Spread sibling limbs around parent using sibling_ind for spatial variation
        # This prevents all recurrent limbs from spawning at the exact same point
        # Keep spread small to maintain visual connection
        spread_factor = (self.sibling_ind - 1) * 0.1  # Reduced spread (was 0.2)
        xyz_1 = self.joint_origin_xyz_1 * 0.3 + spread_factor  # X varies per sibling
        xyz_2 = self.joint_origin_xyz_2 * 0.3 - spread_factor  # Y varies opposite
        # Z offset: place joint at parent's tip + small offset
        # Child will be centered on joint (some visual overlap is fine - they're connected)
        xyz_3 = (parent_len / 2) + 0.05 + self.joint_origin_xyz_3 * 0.1
        xyz = str(xyz_1) + " " + str(xyz_2) + " " + str(xyz_3)
        orig_tag.setAttribute("xyz", xyz)

        joint_tag.appendChild(parent_tag)
        joint_tag.appendChild(child_tag)
        joint_tag.appendChild(axis_tag)
        joint_tag.appendChild(limit_tag)
        joint_tag.appendChild(orig_tag)
        return joint_tag
            



