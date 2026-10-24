"""Synthetic topology tests for the planar CAD wake cut."""

import unittest

import numpy as np

from cad_cut import CutTopologyError, TET_EDGES, split_planar_cut


def make_tet10_mesh():
    points = [
        [0., -1., 0.],   # wake corner, negative span tip
        [1., 1., 0.],    # wake corner, positive span tip
        [0., 0., 0.],    # interior wake corner
        [0., 0., 1.],    # positive-side seed
        [0., 0., -1.],   # negative-side seed
        [0., -1., -2.],  # centroid-straddling tet fourth corner
    ]
    corner_tets = [
        [0, 1, 2, 3],
        [0, 2, 1, 4],
        [0, 1, 3, 5],
    ]
    edge_nodes = {}
    tet10 = []
    for corners in corner_tets:
        row = list(corners)
        for first, second in TET_EDGES:
            edge = tuple(sorted((corners[first], corners[second])))
            if edge not in edge_nodes:
                edge_nodes[edge] = len(points)
                points.append((.5*(np.asarray(points[edge[0]])+np.asarray(points[edge[1]]))).tolist())
            row.append(edge_nodes[edge])
        tet10.append(row)
    wake_face = [0, 1, 2,
                 edge_nodes[(0, 1)], edge_nodes[(1, 2)], edge_nodes[(0, 2)]]
    return np.asarray(points), np.asarray(tet10), np.asarray([wake_face])


class PlanarCutTests(unittest.TestCase):
    def test_star_cut_splits_only_negative_bank_and_keeps_tips_shared(self):
        points, tets, wake = make_tet10_mesh()
        cut = split_planar_cut(points, tets, wake, tip_y=1.)

        self.assertEqual(cut["diagnostics"]["status"], "passed")
        self.assertFalse(cut["diagnostics"]["global_tet_centroid_side_classification_used"])
        self.assertEqual(cut["diagnostics"]["tip_node_count"], 2)
        self.assertEqual(set(cut["tip_nodes"]), {0, 1})
        self.assertEqual(cut["tets"].shape, (3, 10))
        self.assertEqual(cut["points"].shape[0], points.shape[0]+4)
        self.assertTrue(np.array_equal(cut["points"][cut["wake_upper"]],
                                       cut["points"][cut["wake_lower"]]))

        duplicate_by_original = dict(cut["cut_pairs"])
        self.assertEqual(set(duplicate_by_original), set(wake[0])-set(cut["tip_nodes"]))
        self.assertTrue(set(cut["tip_nodes"]).isdisjoint(duplicate_by_original))
        self.assertEqual(cut["wake_upper_tet"].tolist(), [0])
        self.assertEqual(cut["wake_lower_tet"].tolist(), [1])
        self.assertEqual(cut["tets"][1, 0], 0)
        self.assertEqual(cut["tets"][1, 1], duplicate_by_original[2])
        self.assertEqual(cut["tets"][2, 0], 0)
        self.assertEqual(cut["tets"][2, 1], 1)
        self.assertEqual(cut["tets"][2, 4], wake[0, 3])

        straddling_centroid_z = points[tets[2, :4], 2].mean()
        self.assertLess(straddling_centroid_z, 0.)
        star = next(row for row in cut["diagnostics"]["node_stars"]
                    if row["node"] == int(wake[0, 3]))
        self.assertEqual(star["incident_tet_count"], 3)
        self.assertEqual(star["component_tet_counts"], [1, 2])
        self.assertEqual(sorted(star["component_banks"]), [["negative"], ["positive"]])
        self.assertEqual(
            sorted((row["banks"], row["tets"]) for row in star["components"]),
            [(["negative"], [1]), (["positive"], [0, 2])],
        )
        self.assertEqual(star["positive_seed_tets"], [0])
        self.assertEqual(star["negative_seed_tets"], [1])

    def test_wake_face_requires_two_adjacent_tetrahedra(self):
        points, tets, wake = make_tet10_mesh()
        with self.assertRaises(CutTopologyError) as caught:
            split_planar_cut(points, tets[:1], wake, tip_y=1.)
        self.assertIn("exactly two adjacent tetrahedra", str(caught.exception))
        self.assertEqual(caught.exception.diagnostics["status"], "checking")
        self.assertTrue(caught.exception.diagnostics["errors"])

    def test_wake_face_checks_tet10_midside_identity(self):
        points, tets, wake = make_tet10_mesh()
        wake[0, 3] = wake[0, 4]
        with self.assertRaises(CutTopologyError) as caught:
            split_planar_cut(points, tets, wake, tip_y=1.)
        self.assertIn("six distinct node indices", str(caught.exception))

    def test_connectivity_indices_must_be_integer(self):
        points, tets, wake = make_tet10_mesh()
        with self.assertRaises(CutTopologyError) as caught:
            split_planar_cut(points, tets.astype(float), wake, tip_y=1.)
        self.assertIn("integer node indices", str(caught.exception))

    def test_asymmetric_native_span_tip_limits(self):
        points, tets, wake = make_tet10_mesh()
        points[:, 1] += .25
        cut = split_planar_cut(points, tets, wake, tip_y=np.array([-.75, 1.25]))
        self.assertEqual(set(cut["tip_nodes"]), {0, 1})
        self.assertEqual(len(cut["cut_pairs"]), 4)
        self.assertTrue(cut["diagnostics"]["tip_nodes_preserved"])


if __name__ == "__main__":
    unittest.main()
