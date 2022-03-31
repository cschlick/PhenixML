import numpy as np
from sklearn.neighbors import KDTree
import itertools
from itertools import combinations
from collections import defaultdict


class ANIFeaturizer:
    """
    Calculates ANI feature vectors given:

    1. A set of probe parameters
    2. A set of target coordinates (Center of the probe calculations)
    3. A set of interaction atoms (coordinates,elements)

    As currently implemented, it simply ignores elements that do not
    appear in the "elements_considered" parameter. 
    """


    default_params = {
    'radial_cutoff': 4.6,
    'radial_nu': 32,
    'radial_probes': [0.7,
                      1.4,
                      1.9,
                      2.4,
                      3.2,
                      3.8,
                      4.4],
    'angular_cutoff': 3.1,
    'angular_nu': 4,
    'angular_probes': [0.0, 1.57, 3.14, 4.71],
    'angular_radial_probes': [0.7, 1.4, 1.9, 2.4],
    'angular_zeta': 8,
    'min_probed_value': 0.0,
    'exclude_hydrogens': False,
    'elements_considered': ["O", "C", "H", "N", "P", "S", "Cl", "B", "F", "I", "Br"]}


    @classmethod
    def from_fragments_midpoint(cls, fragments, params=None):
        """
        Returns a feature vector calculated at the mean position of a fragment
        """
        if params is None:
            params = ANIFeaturizer.default_params
    
        if not isinstance(fragments, list):
            fragments = [fragments]
    
        # if more than one container present, batch run this function in groups
        n_containers = len(set([frag.mol_container for frag in fragments]))
        if n_containers>1:
            return cls.from_fragments_mixed_containers(fragments,params=params,fragment_method=cls.from_fragments_midpoint)
        
        container = fragments[0].mol_container
        frags_midpoints = np.array([frag.xyz.mean(axis=0) for frag in fragments])
        target_xyz = frags_midpoints
        featurizer = cls(target_xyz, container.xyz, container.elements, ANIFeaturizer.default_params)
        features = featurizer.featurize()
        return features
    
    
    @classmethod
    def from_fragments_atom_centric(cls, fragments, params=None, return_flat=False):
        """
        Return a feature vector calculated at each atom in a fragment. 
    
        Shape of features by default: (n_fragments,n_atoms,n_ani_probes)
        Shape if return_flat=True: (n_fragments,n_atoms*n_ani_probes) (concatenated for each atom in fragment)
        """
    
        if params is None:
            params = ANIFeaturizer.default_params
    
        if not isinstance(fragments, list):
            fragments = [fragments]
        
        # if more than one container present, batch run this function in groups
        n_containers = len(set([frag.mol_container for frag in fragments]))
        if n_containers>1:
            return cls.from_fragments_mixed_containers(fragments,params=params,fragment_method=cls.from_fragments_atom_centric,return_flat=return_flat)
        
        container = fragments[0].mol_container
    
        # get a list of all the atom idxs that apear in any of the fragments
        target_idxs = sorted(list(set(itertools.chain.from_iterable([frag.atom_selection for frag in fragments]))))
        target_xyz = container.xyz[target_idxs]
        featurizer = cls(target_xyz, container.xyz, container.elements, ANIFeaturizer.default_params)
        features = featurizer.featurize()
        idx_mapper = {value: i for i, value in
                      enumerate(target_idxs)}  # a dict to map position container to position in features
        feature_idxs = np.array([idx_mapper[idx] for idx in target_idxs])
        # unpack for each fragment
        fragment_features = []
        for fragment in fragments:
            feature = [features[idx_mapper[i]] for i in fragment.atom_selection]
            fragment_features.append(feature)
        fragment_features = np.array(fragment_features)
    
        if return_flat:
            fragment_features = np.array([np.concatenate(feature) for feature in fragment_features])
        return fragment_features
    
    
    @classmethod
    def from_fragments_mixed_containers(cls, fragments, params=None, fragment_method=None, **kwargs):
        """
        This class is designed to be run with frags from a single environment (ie, a single MolContainer)
    
        If passing a list of fragments from multiple containers, this function:
    
        1. Groups them by container
        2. Featurize each group
        3. Return feature vector in the correct order (corresponding to the index of the fragment list)
        """
        if params is None:
            params = ANIFeaturizer.default_params
    
        if not isinstance(fragments, list):
            fragments = [fragments]
    
        if fragment_method is None:
            fragment_method = ANIFeaturizer.from_fragments_atom_centric
    
        # group fragments into groups of similar containers
        container_groups = defaultdict(list)
        for fragment in fragments:
            container_groups[fragment.mol_container].append(fragment)
    
        # featurize
        feature_groups = defaultdict(list)
        for container, frag_group in container_groups.items():
            features = fragment_method(frag_group,**kwargs)
            feature_groups[container] = features
    
        # unpack in right order
        features_final = []
        for frag in fragments:
            container = frag.mol_container
            idx = container_groups[container].index(frag)
            feature = feature_groups[container][idx]
            features_final.append(feature)
        features_final = np.array(features_final)
    
        # A test
        # For me, the simpler method below is around 3X slower than above
    
        # features_test = []
        # for frag in fragments:
        #     feature = fragment_method(frag)[0]
        #     features_test.append(feature)
        # features_test = np.array(features_test)
    
        # assert np.all(features_test==features_final)
    
        return features_final

    
    def __init__(self, target_xyz, interaction_xyz, interaction_elements, params, calc_mode="optimized", debug=False):
        self.params = params
        self.target_xyz = target_xyz
    
        self.n_targets = len(target_xyz)
        self.interaction_inds = np.array(
            [i for i, element in enumerate(interaction_elements) if element in self.params["elements_considered"]])
        self.interaction_elements = [interaction_elements[i] for i in self.interaction_inds]
        self.interaction_xyz = interaction_xyz[self.interaction_inds]
    
        self.calc_mode = calc_mode
        self.debug = debug
    
        # set parameters
        self.rc_rad = self.params["radial_cutoff"]
        self.nu_rad = self.params["radial_nu"]
        self.rs_rad = np.array(self.params["radial_probes"])
        self.rc_ang = self.params["angular_cutoff"]
        self.rs_ang = np.array(self.params["angular_radial_probes"])
        self.ts = np.array(self.params["angular_probes"])
        self.nu_ang = self.params["angular_nu"]
        self.zeta = self.params["angular_zeta"]
    
        self.radial_elements = sorted(set(self.params["elements_considered"]))
        self.angular_elements = [ij for ij in list(combinations(self.radial_elements, 2))] + [(e, e) for e in
                                                                                              self.radial_elements]
        self.angular_elements = [tuple(sorted(ij)) for ij in self.angular_elements]
    
    
    def featurize(self):
        gr, ga = self.gr, self.ga
        return np.hstack([gr, ga])
    
    
    @property
    def ga(self):
        if not hasattr(self, "_ga"):
            if self.calc_mode == "simple":
                _ga = self.calc_angular_simple()
            elif self.calc_mode == "optimized":
                _ga = self.calc_angular_optimized()
        return _ga
    
    
    @property
    def gr(self):
        if not hasattr(self, "_gr"):
            if self.calc_mode == "simple":
                _gr = self.calc_radial_simple()
            elif self.calc_mode == "optimized":
                _gr = self.calc_radial_optimized()
        return _gr
    
    
    @property
    def kdtree_interact(self):
        if not hasattr(self, "_kdtree_interact"):
            self._kdtree_interact = KDTree(self.interaction_xyz)
        return self._kdtree_interact
    
    
    @staticmethod
    def fc_vect(rij, rc):
        return np.piecewise(rij, [rij <= rc, rij > rc],
                            [lambda x: (0.5 * np.cos((np.pi * x) / rc) + 0.5), 0])
    
    
    def calc_radial_simple(self, return_flat=True, debug=False):
        n_targets = self.n_targets
        n_rad = len(self.radial_elements)
        n_rs = len(self.rs_rad)
        if debug:
            self.ij_inds = []  # debug
        radial_vector = np.zeros((n_targets, n_rad, n_rs))
        for i, i_xyz in enumerate(self.target_xyz):
            for j, element_j in enumerate(self.interaction_elements):
                j_xyz = self.interaction_xyz[j]
                rij = np.linalg.norm(i_xyz - j_xyz)
                if rij > 0:
                    gr = np.zeros(n_rs)
    
                    if rij > self.rc_rad:
                        pass  # set all to zero
                    else:
                        fc = 0.5 * np.cos((np.pi * rij) / self.rc_rad) + 0.5
    
                        for r, rs in enumerate(self.rs_rad):
                            gr[r] += np.exp(-self.nu_rad * (rij - rs) ** 2) * fc
    
                        if debug:
                            self.ij_inds.append([i, j])  # debug
    
                    e = self.radial_elements.index(element_j)
                    radial_vector[i, e] += gr
    
        if not return_flat:
            return radial_vector
        else:
            return radial_vector.reshape((n_targets, n_rad * n_rs))
    
    
    def calc_radial_optimized(self, return_flat=True):
        rc_rad = self.rc_rad
        n_targets = self.n_targets
        n_rs = len(self.rs_rad)
        n_rad = len(self.radial_elements)
    
        inds, dists = self.kdtree_interact.query_radius(self.target_xyz, self.rc_rad, return_distance=True)
        ij_inds = np.array([(i, j) for i, ind in enumerate(inds) for j_idx, j in enumerate(ind) if dists[i][j_idx] > 0])
    
        gr = np.zeros((n_targets, n_rad, n_rs))
        if len(ij_inds) > 0:
    
            nu_rad = self.nu_rad
            rs_rad = self.rs_rad
    
            rad_i = ij_inds[:, 0]
            rad_j = ij_inds[:, 1]
            rad_xyz_i = self.target_xyz[rad_i]
            rad_xyz_j = self.interaction_xyz[rad_j]
    
            rad_xyz_ij = rad_xyz_j - rad_xyz_i
    
            rad_xyz_ij = rad_xyz_j - rad_xyz_i
            rad_rij = np.linalg.norm(rad_xyz_ij, axis=1)
    
            fc = self.fc_vect(rad_rij, self.rc_rad)
    
            radial = np.exp(-self.nu_rad * (rad_rij[:, np.newaxis] - self.rs_rad) ** 2) * fc[:, np.newaxis]
    
            for n, (i, j) in enumerate(ij_inds):
                ej = self.interaction_elements[j]
                component_index = self.radial_elements.index(ej)
                gr[i, component_index] += radial[n]
        if return_flat:
            gr = gr.reshape((n_targets, n_rad * n_rs))
        return gr
    
    
    def calc_angular_simple(self, return_flat=True, debug=True):
        n_targets = self.n_targets
        n_interactions = len(self.interaction_inds)
        n_ang = len(self.angular_elements)
        n_rs = len(self.rs_ang)
        n_ts = len(self.ts)
    
        if debug:
            self.debug_dict_ang = {}
            self.ijk_triples = []
        angular_vector = np.zeros((n_targets, n_ang, n_ts, n_rs))
    
        for i, i_xyz in enumerate(self.target_xyz):
            for j, element_j in enumerate(self.interaction_elements):
                for k, element_k in enumerate(self.interaction_elements):
                    if j != k:
                        j_xyz = self.interaction_xyz[j]
                        k_xyz = self.interaction_xyz[k]
                        rij = np.linalg.norm(i_xyz - j_xyz)
                        rik = np.linalg.norm(i_xyz - k_xyz)
                        if rij > 0 and rik > 0:
    
                            ga = np.zeros((n_ts, n_rs))
                            if rij > self.rc_ang or rik > self.rc_ang:
                                pass
                            else:
                                self.ijk_triples.append([i, j, k])
                                fcij = 0.5 * np.cos((np.pi * rij) / self.rc_ang) + 0.5
                                fcik = 0.5 * np.cos((np.pi * rik) / self.rc_ang) + 0.5
    
                                for t, ts in enumerate(self.ts):
                                    # i,j,k == b,a,c
                                    ba = j_xyz - i_xyz
                                    bc = k_xyz - i_xyz
                                    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                                    angle_tol = 1e-4
                                    if cosine_angle > 1:
                                        assert (cosine_angle - 1 < angle_tol)
                                        cosine_angle = 1.0
                                    elif cosine_angle < -1:
                                        assert (cosine_angle + 1 < angle_tol)
                                        cosine_angle = -1.0
    
                                    tijk = np.arccos(cosine_angle)
                                    angular_term = (1 + np.cos(tijk - ts)) ** self.zeta
    
                                    for r, rs in enumerate(self.rs_ang):
                                        radial_term = np.exp(-self.nu_ang * (((rij + rik) / 2) - rs) ** 2)
                                        ga[t, r] = angular_term * radial_term * fcij * fcik
    
                                    # match to an angular component
                                    element_pair = tuple(sorted([element_j, element_k]))
                                    if element_pair in self.angular_elements:
                                        a = self.angular_elements.index(element_pair)
                                    else:
                                        print("ERROR: j,k element pair not in full element list")
    
                                        # debug
                                        if debug:
                                            key = (i, j, k, ts, rs)
                                            assert (key not in self.debug_dict_ang)
                                            self.debug_dict_ang[key] = {}
                                            self.debug_dict_ang[key]["tijk"] = tijk
                                            self.debug_dict_ang[key]["rij"] = rij
                                            self.debug_dict_ang[key]["rik"] = rik
                                            self.debug_dict_ang[key]["fcij"] = fcij
                                            self.debug_dict_ang[key]["fcik"] = fcik
                                            self.debug_dict_ang[key]["angular_term"] = angular_term
                                            self.debug_dict_ang[key]["radial_term"] = radial_term
                                            self.debug_dict_ang[key]["combined"] = angular_term * radial_term * fcij * fcik
                                            self.debug_dict_ang[key]["combined_scaled"] = 2 ** (1 - self.zeta) * (
                                                    angular_term * radial_term * fcij * fcik)
                                            self.debug_dict_ang[key]["ek"] = element_pair[0] + "." + element_pair[1]
    
                            angular_vector[i, a] += ga
    
        angular_vector *= 2 ** (1 - self.zeta)
        if return_flat:
            return angular_vector.reshape((n_targets, n_ang * n_rs * n_ts))
        else:
            return angular_vector
    
    
    def calc_angular_optimized(self, return_flat=True, debug=True):
        n_targets = self.n_targets
        n_interactions = len(self.interaction_inds)
        n_ang = len(self.angular_elements)
        n_rs_ang = len(self.rs_ang)
        n_ts = len(self.ts)
        rc_ang = self.rc_ang
    
        n_ang = len(self.angular_elements)
    
        inds, dists = self.kdtree_interact.query_radius(self.target_xyz, rc_ang, return_distance=True)
    
        # new_inds will remove the self pairs
        new_inds = []
        for i, ind in enumerate(inds):
            new_ind = []
            for j, n in enumerate(ind):
                d = dists[i][j]
                if d > 0:
                    new_ind.append(n)
            new_inds.append(new_ind)
    
        ijk_inds = [(i, j, k) for i, ind in enumerate(new_inds) for j, k in combinations(ind, 2) if j != k]
        ijk_inds += [(i, k, j) for i, ind in enumerate(new_inds) for j, k in combinations(ind, 2) if j != k]
        ijk_inds = np.array(ijk_inds)
        # Note: The above will do redundant calculations. Calculate every i,j,k and i,k,j pair. This is
        #       done for consistency with previous code and to make debugging easier
    
        if debug:
            self.ijk_inds = ijk_inds
        ga = np.zeros((n_targets, n_ang, n_ts, n_rs_ang))
    
        if len(ijk_inds) > 0:
            xyz_i = self.target_xyz[ijk_inds[:, 0]]
            xyz_j = self.interaction_xyz[ijk_inds[:, 1]]
            xyz_k = self.interaction_xyz[ijk_inds[:, 2]]
    
            xyz_ij = xyz_j - xyz_i
            xyz_ik = xyz_k - xyz_i
    
            rij = np.linalg.norm(xyz_ij, axis=1)
            rik = np.linalg.norm(xyz_ik, axis=1)
    
            # calculate the angle between ijk sites, centered on site i
            cosine_angle = np.einsum('ij,ij->i', xyz_ij, xyz_ik) / (rij * rik)
            neg_inds, = np.where(cosine_angle < -1)
            pos_inds, = np.where(cosine_angle > 1)
            cosine_angle[neg_inds] = -1.0
            cosine_angle[pos_inds] = 1.0
            tijk = np.arccos(cosine_angle)
    
            # calculate angular and modified radial components of the environment vector (see Equation 4)
            angular = (1 + np.cos(tijk[:, np.newaxis] - self.ts)) ** self.zeta
            radial = np.exp(-self.nu_ang * (((rij + rik) / 2)[:, np.newaxis] - self.rs_ang) ** 2)
    
            # calculate the radial weights (see Equation 2)
            fcij = self.fc_vect(rij, self.rc_ang)
            fcik = self.fc_vect(rik, self.rc_ang)
    
            # multiply all pieces, broadcasts to shape (l,m,n) where l indexes the triples,
            # m the ts values, n the rs values
            ang_mult = angular[:, :, np.newaxis] * radial[:, np.newaxis, :] * \
                       fcij[:, np.newaxis, np.newaxis] * fcik[:, np.newaxis, np.newaxis]
    
            for n, (i, j, k) in enumerate(ijk_inds):
                ek = tuple(sorted([self.interaction_elements[j], self.interaction_elements[k]]))
                component_index = self.angular_elements.index(ek)
                ga[i, component_index] += ang_mult[n]
        if return_flat:
            ga = ga.reshape((n_targets, n_ang * n_ts * n_rs_ang))
        ga *= (2 ** (1 - self.zeta))
    
        if debug:
            for key, value in locals().items():
                key = "debug_ang_" + key
                setattr(self, key, value)
        return ga
