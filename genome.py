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
    def get_gene_spec():
        gene_spec =  {"link-shape":{"scale":1},
            "link-length": {"scale":2},  # Increased for longer, more visible limbs
            "link-radius": {"scale":0.4},  # Reduced for thinner limbs (less overlap)
            "link-recurrence": {"scale":4},  # Increased for more limbs per gene
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
    def crossover(g1, g2):
        x1 = random.randint(0, len(g1)-1)
        x2 = random.randint(0, len(g2)-1)
        # Combine genes from both parents as a list (not numpy array)
        # Each gene is a numpy array, genome is a list of genes
        g3 = list(g1[x1:]) + list(g2[x2:])
        if len(g3) > len(g1):
            g3 = g3[0:len(g1)]
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
    def shrink_mutate(genome, rate):
        if len(genome) == 1:
            # Deep copy to prevent corruption of parent DNA
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

        # Determine shape based on link_shape gene value (0-1 range)
        # 0.0 - 0.33: cylinder, 0.33 - 0.66: sphere, 0.66 - 1.0: box
        if self.link_shape <= 0.33:
            # Cylinder shape
            shape_tag = adom.createElement("cylinder")
            shape_tag.setAttribute("length", str(actual_length))
            shape_tag.setAttribute("radius", str(actual_radius))
            # Mass = density * volume = pi * r^2 * height
            mass = np.pi * (actual_radius * actual_radius) * actual_length
        elif self.link_shape <= 0.66:
            # Sphere shape - use radius based on average of length and radius
            sphere_radius = (actual_length + actual_radius) / 2
            shape_tag = adom.createElement("sphere")
            shape_tag.setAttribute("radius", str(sphere_radius))
            # Mass = density * volume = (4/3) * pi * r^3
            mass = (4.0/3.0) * np.pi * (sphere_radius ** 3)
        else:
            # Box shape - use length and radius to define box dimensions
            shape_tag = adom.createElement("box")
            # Box size: width=2*radius, depth=2*radius, height=length
            box_size = str(actual_radius*2) + " " + str(actual_radius*2) + " " + str(actual_length)
            shape_tag.setAttribute("size", box_size)
            # Mass = density * volume = width * depth * height
            mass = (actual_radius * 2) * (actual_radius * 2) * actual_length

        geom_tag.appendChild(shape_tag)
        vis_tag.appendChild(geom_tag)

        coll_tag = adom.createElement("collision")
        c_geom_tag = adom.createElement("geometry")

        # Create collision shape (same as visual)
        if self.link_shape <= 0.33:
            c_shape_tag = adom.createElement("cylinder")
            c_shape_tag.setAttribute("length", str(actual_length))
            c_shape_tag.setAttribute("radius", str(actual_radius))
        elif self.link_shape <= 0.66:
            sphere_radius = (actual_length + actual_radius) / 2
            c_shape_tag = adom.createElement("sphere")
            c_shape_tag.setAttribute("radius", str(sphere_radius))
        else:
            c_shape_tag = adom.createElement("box")
            box_size = str(actual_radius*2) + " " + str(actual_radius*2) + " " + str(actual_length)
            c_shape_tag.setAttribute("size", box_size)

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
        # Calculate proper inertia based on shape type
        if self.link_shape <= 0.33:
            # Cylinder inertia
            # Ixx = Iyy = (1/12) * m * (3*r^2 + h^2)
            # Izz = (1/2) * m * r^2
            ixx = (1.0/12.0) * mass * (3 * actual_radius * actual_radius + actual_length * actual_length)
            iyy = ixx
            izz = (1.0/2.0) * mass * actual_radius * actual_radius
        elif self.link_shape <= 0.66:
            # Sphere inertia (solid sphere)
            # Ixx = Iyy = Izz = (2/5) * m * r^2
            sphere_radius = (actual_length + actual_radius) / 2
            ixx = (2.0/5.0) * mass * sphere_radius * sphere_radius
            iyy = ixx
            izz = ixx
        else:
            # Box inertia (solid cuboid)
            # Ixx = (1/12) * m * (h^2 + d^2)
            # Iyy = (1/12) * m * (w^2 + h^2)
            # Izz = (1/12) * m * (w^2 + d^2)
            w = actual_radius * 2
            d = actual_radius * 2
            h = actual_length
            ixx = (1.0/12.0) * mass * (h*h + d*d)
            iyy = (1.0/12.0) * mass * (w*w + h*h)
            izz = (1.0/12.0) * mass * (w*w + d*d)
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
        # Position joint at end of parent link to prevent floating/detached limbs
        # URDF cylinders extend along Z-axis, so offset must be in Z (xyz_3)
        # Use parent's link_length (not child's) to position at parent's endpoint
        # Enforce minimum parent length for joint calculation
        parent_len = max(self.parent_link_length, 0.1)
        # Spread sibling limbs around parent using sibling_ind for spatial variation
        # This prevents all recurrent limbs from spawning at the exact same point
        spread_factor = (self.sibling_ind - 1) * 0.2  # Spread siblings apart
        xyz_1 = self.joint_origin_xyz_1 * 0.5 + spread_factor  # X varies per sibling
        xyz_2 = self.joint_origin_xyz_2 * 0.5 - spread_factor  # Y varies opposite
        xyz_3 = parent_len * 0.5 + self.joint_origin_xyz_3 * 0.3  # Z at parent endpoint
        xyz = str(xyz_1) + " " + str(xyz_2) + " " + str(xyz_3)
        orig_tag.setAttribute("xyz", xyz)

        joint_tag.appendChild(parent_tag)
        joint_tag.appendChild(child_tag)
        joint_tag.appendChild(axis_tag)
        joint_tag.appendChild(limit_tag)
        joint_tag.appendChild(orig_tag)
        return joint_tag
            



