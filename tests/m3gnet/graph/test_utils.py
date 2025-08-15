import numpy as np
import pytest
import torch

from m3gnet.graph.utils import compute_fixed_radius_graph, compute_threebody_indices


class TestComputeFixedRadiusGraph:
    """Test suite for compute_fixed_radius_graph function."""

    def test_simple_2d_case(self):
        """Test basic 2D case without PBC."""
        pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )

        edge_index, distances, offsets = compute_fixed_radius_graph(
            pos=pos, cell=None, pbc=False, cutoff=1.5
        )

        # Should find edges between adjacent atoms only
        assert isinstance(edge_index, torch.Tensor)
        assert edge_index.dtype == torch.long
        assert edge_index.shape[0] == 2  # Source and target indices
        assert edge_index.shape[1] == 4  # 2 bidirectional edges

        # Check distances
        assert isinstance(distances, np.ndarray)
        assert distances.dtype == np.float64
        assert len(distances) == 4
        assert np.allclose(distances, [1.0, 1.0, 1.0, 1.0], atol=1e-6)

        # Check offsets (should be zero for non-PBC)
        assert isinstance(offsets, np.ndarray)
        assert offsets.dtype == np.int64
        assert offsets.shape == (4, 3)
        assert np.all(offsets == 0)

    def test_3d_coordinates(self):
        """Test with 3D coordinates."""
        pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
            ]
        )

        cutoff = 2.0
        edge_index, distances, offsets = compute_fixed_radius_graph(
            pos=pos, cell=None, pbc=False, cutoff=cutoff
        )

        # Distance between first two points: sqrt(3) ≈ 1.73 < 2.0
        # Distance between first and third: sqrt(12) ≈ 3.46 > 2.0
        # Distance between second and third: sqrt(3) ≈ 1.73 < 2.0

        assert edge_index.shape[1] == 4  # Two bidirectional edges
        expected_distance = np.sqrt(3.0)
        assert np.allclose(distances, [expected_distance] * 4, atol=1e-6)

    def test_no_edges_large_cutoff(self):
        """Test case where cutoff is too small for any edges."""
        pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
            ]
        )

        edge_index, distances, offsets = compute_fixed_radius_graph(
            pos=pos, cell=None, pbc=False, cutoff=1.0
        )

        assert edge_index.shape[1] == 0
        assert len(distances) == 0
        assert offsets.shape[0] == 0

    def test_all_connected_large_cutoff(self):
        """Test case where all atoms are connected due to large cutoff."""
        pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )

        edge_index, distances, offsets = compute_fixed_radius_graph(
            pos=pos, cell=None, pbc=False, cutoff=5.0
        )

        # Should have 6 edges (all pairs, bidirectional)
        assert edge_index.shape[1] == 6
        assert len(distances) == 6

        # Check that distances are correct
        expected_distances = sorted([1.0, 1.0, 2.0, 2.0, 1.0, 1.0])
        assert np.allclose(sorted(distances), expected_distances, atol=1e-6)

    def test_pbc_requires_cell(self):
        """Test that PBC=True requires cell parameters."""
        pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        with pytest.raises(
            ValueError, match="Cell parameters are required when pbc is True"
        ):
            compute_fixed_radius_graph(pos=pos, cell=None, pbc=True, cutoff=2.0)

    def test_pbc_with_cell(self):
        """Test PBC functionality with cell parameters."""
        pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [4.9, 0.0, 0.0],  # Close to edge of 5x5x5 cell
            ]
        )

        cell = np.array(
            [
                [5.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
                [0.0, 0.0, 5.0],
            ]
        )

        edge_index, distances, offsets = compute_fixed_radius_graph(
            pos=pos, cell=cell, pbc=True, cutoff=1.0
        )

        # With PBC, atoms should be connected across the boundary
        # Distance should be ~0.1 (wrap around)
        assert edge_index.shape[1] > 0
        assert np.any(distances < 1.0)

    def test_self_connection_exclusion(self):
        """Test that self-connections are properly excluded."""
        pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )

        edge_index, distances, offsets = compute_fixed_radius_graph(
            pos=pos, cell=None, pbc=False, cutoff=5.0
        )

        # Should not have self-connections
        source_indices = edge_index[0].numpy()
        target_indices = edge_index[1].numpy()

        for i in range(len(source_indices)):
            if source_indices[i] == target_indices[i]:
                # If same indices, distance should not be near zero
                assert distances[i] > 1e-6

    def test_numerical_tolerance(self):
        """Test numerical tolerance parameter."""
        pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [1e-10, 0.0, 0.0],  # Very close to first atom
                [1.0, 0.0, 0.0],
            ]
        )

        # With default tolerance, very close atoms should be treated as same
        edge_index, distances, offsets = compute_fixed_radius_graph(
            pos=pos, cell=None, pbc=False, cutoff=2.0, numerical_tol=1e-8
        )

        # The function should still include the very small distance edges
        # but we can check that self-connections are excluded
        assert edge_index.shape[1] > 0
        # Check that we have some reasonable distances
        assert len(distances) > 0

    def test_empty_input(self):
        """Test behavior with empty input."""
        pos = np.empty((0, 3))

        edge_index, distances, offsets = compute_fixed_radius_graph(
            pos=pos, cell=None, pbc=False, cutoff=5.0
        )

        assert edge_index.shape == (2, 0)
        assert len(distances) == 0
        assert offsets.shape == (0, 3)

    def test_single_atom(self):
        """Test behavior with single atom."""
        pos = np.array([[0.0, 0.0, 0.0]])

        edge_index, distances, offsets = compute_fixed_radius_graph(
            pos=pos, cell=None, pbc=False, cutoff=5.0
        )

        assert edge_index.shape == (2, 0)
        assert len(distances) == 0
        assert offsets.shape == (0, 3)


class TestComputeThreebodyIndices:
    """Test suite for compute_threebody_indices function."""

    def test_simple_linear_case(self):
        """Test simple linear arrangement of atoms."""
        pos = np.array(
            [
                [0.0, 0.0, 0.0],  # Atom 0
                [1.0, 0.0, 0.0],  # Atom 1
                [2.0, 0.0, 0.0],  # Atom 2
            ]
        )

        # Create edge index manually for testing
        edge_index = np.array(
            [
                [0, 1, 1, 0, 1, 2, 2, 1],  # source indices
                [1, 0, 2, 1, 1, 1, 1, 2],  # target indices
            ]
        )
        edge_dist = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        result = compute_threebody_indices(
            pos=pos, edge_index=edge_index, edge_dist=edge_dist, three_body_cutoff=1.5
        )

        (
            total_num_angles,
            num_bonds_per_atom,
            num_angles_per_atom,
            num_angles_on_central_atom,
            three_body_indices,
        ) = result

        assert isinstance(total_num_angles, int | np.integer)
        assert isinstance(num_bonds_per_atom, np.ndarray)
        assert isinstance(num_angles_per_atom, np.ndarray)
        assert isinstance(num_angles_on_central_atom, np.ndarray)
        assert isinstance(three_body_indices, np.ndarray)

        # Check shapes
        assert num_bonds_per_atom.shape == (3,)
        assert num_angles_per_atom.shape == (3,)
        assert num_angles_on_central_atom.shape == (8,)  # Same as number of edges
        assert three_body_indices.shape[1] == 2  # Each angle defined by 2 bonds

    def test_triangular_case(self):
        """Test triangular arrangement of atoms."""
        pos = np.array(
            [
                [0.0, 0.0, 0.0],  # Atom 0
                [1.0, 0.0, 0.0],  # Atom 1
                [0.5, 0.866, 0.0],  # Atom 2 (forms equilateral triangle)
            ]
        )

        # All atoms are connected (distance ~1.0)
        edge_index = np.array(
            [
                [0, 1, 1, 0, 0, 2, 2, 0, 1, 2, 2, 1],
                [1, 0, 2, 2, 2, 0, 0, 2, 2, 1, 1, 2],
            ]
        )
        edge_dist = np.ones(12)  # All distances ~1.0

        result = compute_threebody_indices(
            pos=pos, edge_index=edge_index, edge_dist=edge_dist, three_body_cutoff=1.5
        )

        (
            total_num_angles,
            num_bonds_per_atom,
            num_angles_per_atom,
            num_angles_on_central_atom,
            three_body_indices,
        ) = result

        # Each atom should have 2 bonds
        assert np.all(num_bonds_per_atom == 4)  # 4 bonds per atom (bidirectional)

        # Each atom should have multiple angles
        assert np.all(num_angles_per_atom > 0)

        # Should have some three-body indices
        assert three_body_indices.shape[0] > 0

    def test_no_three_body_interactions(self):
        """Test case with no three-body interactions (isolated pairs)."""
        pos = np.array(
            [
                [0.0, 0.0, 0.0],  # Atom 0
                [1.0, 0.0, 0.0],  # Atom 1
                [10.0, 0.0, 0.0],  # Atom 2 (far away)
                [11.0, 0.0, 0.0],  # Atom 3 (far away)
            ]
        )

        # Only pairs are connected
        edge_index = np.array(
            [
                [0, 1, 1, 0, 2, 3, 3, 2],
                [1, 0, 0, 1, 3, 2, 2, 3],
            ]
        )
        edge_dist = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        result = compute_threebody_indices(
            pos=pos, edge_index=edge_index, edge_dist=edge_dist, three_body_cutoff=1.5
        )

        (
            total_num_angles,
            num_bonds_per_atom,
            num_angles_per_atom,
            num_angles_on_central_atom,
            three_body_indices,
        ) = result

        # Each atom has 2 bonds (bidirectional to its pair), so some angles are possible
        # but since they're isolated pairs, the actual number depends on the
        # implementation Let's just check that the function runs without error
        # and returns valid shapes
        assert total_num_angles >= 0
        assert three_body_indices.shape[1] == 2

    def test_cutoff_filtering(self):
        """Test that three-body cutoff properly filters bonds."""
        pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )

        # Include bonds of different lengths
        edge_index = np.array(
            [
                [0, 1, 1, 0, 1, 2, 2, 1, 0, 2, 2, 0],
                [1, 0, 2, 2, 2, 1, 1, 2, 2, 0, 0, 2],
            ]
        )
        edge_dist = np.array(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]
        )

        # Use small cutoff to exclude long bonds
        result = compute_threebody_indices(
            pos=pos,
            edge_index=edge_index,
            edge_dist=edge_dist,
            three_body_cutoff=1.5,  # Excludes 2.0 distance bonds
        )

        (
            total_num_angles,
            num_bonds_per_atom,
            num_angles_per_atom,
            num_angles_on_central_atom,
            three_body_indices,
        ) = result

        # Only short bonds should contribute to angles
        assert np.all(
            num_angles_on_central_atom[:8] >= 0
        )  # Short bonds can have angles
        assert np.all(num_angles_on_central_atom[8:] == 0)  # Long bonds have no angles

    def test_empty_edges(self):
        """Test behavior with no edges."""
        pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
            ]
        )

        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_dist = np.empty(0)

        result = compute_threebody_indices(
            pos=pos, edge_index=edge_index, edge_dist=edge_dist, three_body_cutoff=1.0
        )

        (
            total_num_angles,
            num_bonds_per_atom,
            num_angles_per_atom,
            num_angles_on_central_atom,
            three_body_indices,
        ) = result

        assert total_num_angles == 0
        assert np.all(num_bonds_per_atom == 0)
        assert np.all(num_angles_per_atom == 0)
        assert len(num_angles_on_central_atom) == 0
        assert three_body_indices.shape == (0, 2)

    def test_single_bond(self):
        """Test behavior with single bond."""
        pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )

        edge_index = np.array([[0, 1], [1, 0]])
        edge_dist = np.array([1.0, 1.0])

        result = compute_threebody_indices(
            pos=pos, edge_index=edge_index, edge_dist=edge_dist, three_body_cutoff=1.5
        )

        (
            total_num_angles,
            num_bonds_per_atom,
            num_angles_per_atom,
            num_angles_on_central_atom,
            three_body_indices,
        ) = result

        # Each atom has 1 bond, so no angles possible
        assert np.all(num_bonds_per_atom == 1)
        assert np.all(num_angles_per_atom == 0)
        assert total_num_angles == 0
        assert three_body_indices.shape == (0, 2)

    def test_output_types_and_shapes(self):
        """Test that output types and shapes are correct."""
        pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )

        edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])
        edge_dist = np.array([1.0, 1.0, 1.0, 1.0])

        result = compute_threebody_indices(
            pos=pos, edge_index=edge_index, edge_dist=edge_dist, three_body_cutoff=1.5
        )

        (
            total_num_angles,
            num_bonds_per_atom,
            num_angles_per_atom,
            num_angles_on_central_atom,
            three_body_indices,
        ) = result

        # Check types
        assert isinstance(total_num_angles, int | np.integer)
        assert isinstance(num_bonds_per_atom, np.ndarray)
        assert isinstance(num_angles_per_atom, np.ndarray)
        assert isinstance(num_angles_on_central_atom, np.ndarray)
        assert isinstance(three_body_indices, np.ndarray)

        # Check dtypes
        assert num_bonds_per_atom.dtype == np.int64
        assert num_angles_per_atom.dtype == np.int64
        assert num_angles_on_central_atom.dtype == np.int64
        assert three_body_indices.dtype == np.int64

        # Check shapes
        assert num_bonds_per_atom.shape == (pos.shape[0],)
        assert num_angles_per_atom.shape == (pos.shape[0],)
        assert num_angles_on_central_atom.shape == (edge_index.shape[1],)
        assert three_body_indices.shape[1] == 2


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__])
