import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from m3gnet.graph.converter import GraphConverter


class TestGraphConverter:
    """Test suite for GraphConverter."""

    @pytest.fixture
    def simple_converter(self):
        """Create a simple GraphConverter instance."""
        return GraphConverter(cutoff=2.5, pbc=False)

    @pytest.fixture
    def large_cutoff_converter(self):
        """Create a converter with larger cutoff for more edges."""
        return GraphConverter(cutoff=6.0, pbc=False)

    @pytest.fixture
    def pbc_converter(self):
        """Create a converter with periodic boundary conditions."""
        return GraphConverter(cutoff=5.0, pbc=True, three_body_cutoff=3.0)

    @pytest.fixture
    def three_body_converter(self):
        """Create a converter with three-body interactions enabled."""
        return GraphConverter(
            cutoff=3.0,
            pbc=False,
            three_body_cutoff=2.0,
        )

    def test_simple_unbatched_graph(self, simple_converter):
        """
        Tests graph creation for a single, simple structure (no batching).
        """
        # 1. Arrange
        # Points: A=(0,0,0), B=(2,0,0), C=(5,0,0)
        pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ]
        )
        cell = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])

        # 2. Act
        data = simple_converter.convert(pos=pos, cell=cell)

        # 3. Assert
        # We expect an edge between A and B, and B and A (undirected)
        # The distance is sqrt((2-0)^2) = 2.0, which is < 2.5
        # The distance between A and C is 5.0, which is > 2.5

        assert isinstance(data, Data)
        assert data.num_nodes == 3
        assert data.num_edges == 2

        # Check that edge distances are correct
        expected_distances = [2.0, 2.0]  # A-B and B-A distances
        assert np.allclose(data.edge_dist, expected_distances, atol=1e-6)

        # Check that positions are preserved
        assert np.allclose(data.pos, pos)

    def test_no_edges_with_small_cutoff(self):
        """Test that no edges are created when cutoff is too small."""
        # Points: A=(0,0,0), B=(3,0,0), C=(6,0,0)
        pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
            ]
        )
        cell = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])

        converter = GraphConverter(cutoff=1.0, pbc=False)  # Very small cutoff
        data = converter.convert(pos=pos, cell=cell)

        assert data.num_nodes == 3
        assert data.num_edges == 0
        assert data.edge_index.shape[1] == 0

    def test_all_edges_with_large_cutoff(self, large_cutoff_converter):
        """Test that all possible edges are created with large cutoff."""
        # Points: A=(0,0,0), B=(1,0,0), C=(2,0,0)
        pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )
        cell = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])

        data = large_cutoff_converter.convert(pos=pos, cell=cell)

        # With cutoff 6.0, all atoms should be connected
        # Expected: 6 edges (A-B, B-A, A-C, C-A, B-C, C-B)
        assert data.num_nodes == 3
        assert data.num_edges == 6

        # Sort edge attributes to ensure consistent ordering
        sorted_edge_dist = np.sort(data.edge_dist)

        # Check that all distances are correct
        expected_distances = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0]
        assert np.allclose(sorted_edge_dist, expected_distances, atol=1e-6)

    def test_3d_coordinates(self, simple_converter):
        """Test graph creation with 3D coordinates."""
        # Points in 3D space
        pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [3.0, 3.0, 3.0],
            ]
        )
        cell = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])

        data = simple_converter.convert(pos=pos, cell=cell)

        # Distance A-B = sqrt(1² + 1² + 1²) = sqrt(3) ≈ 1.73 < 2.5
        # Distance A-C = sqrt(3² + 3² + 3²) = sqrt(27) ≈ 5.20 > 2.5
        # Distance B-C = sqrt(2² + 2² + 2²) = sqrt(12) ≈ 3.46 > 2.5

        assert data.num_nodes == 3
        assert data.num_edges == 2  # Only A-B connection

        # Check that the edge distance is approximately sqrt(3)
        expected_distance = np.sqrt(3.0)
        assert np.allclose(
            data.edge_dist,
            [expected_distance, expected_distance],
            atol=1e-6,
        )

    def test_edge_index_format(self, simple_converter):
        """Test that edge_index has correct format."""
        pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        cell = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])

        data = simple_converter.convert(pos=pos, cell=cell)

        # Should have 2 edges (A->B and B->A)
        assert data.edge_index.shape == (2, 2)
        assert data.edge_index.dtype == torch.long

        # Check that edge_index contains valid node indices
        assert torch.all(data.edge_index >= 0)
        assert torch.all(data.edge_index < data.num_nodes)

    def test_edge_dist_format(self, simple_converter):
        """Test that edge_dist has correct format."""
        pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        cell = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])

        data = simple_converter.convert(pos=pos, cell=cell)

        # Should have 2 edge attributes (distances)
        assert data.edge_dist.shape == (2,)
        assert data.edge_dist.dtype == torch.float32

        # All distances should be positive
        assert torch.all(data.edge_dist > 0)

    def test_three_body_converter_creation(self, three_body_converter):
        """Test that three-body converter is created correctly."""
        assert three_body_converter.cutoff == 3.0
        assert three_body_converter.three_body_cutoff == 2.0
        assert three_body_converter.pbc is False

    def test_three_body_converter_forward(self, three_body_converter):
        """Test three-body converter forward pass."""
        pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )
        cell = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])

        data = three_body_converter.convert(pos=pos, cell=cell)

        assert isinstance(data, Data)
        assert data.num_nodes == 3
        # Should create edges based on cutoff
        assert data.num_edges > 0

        # Check three-body specific attributes
        assert hasattr(data, "three_body_indices")
        assert hasattr(data, "num_bonds_per_atom")
        assert hasattr(data, "num_angles_per_atom")
        assert hasattr(data, "num_angles_on_central_atom")
        assert hasattr(data, "total_num_angles")

    def test_different_cutoff_values(self):
        """Test converter behavior with different cutoff values."""
        pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        )
        cell = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])

        # Test with cutoff = 1.5 (should only connect adjacent atoms)
        converter_small = GraphConverter(cutoff=1.5, pbc=False)
        data_small = converter_small.convert(pos=pos, cell=cell)
        assert data_small.num_edges == 3 * 2  # Only 0-1, 1-2, 2-3 connections

        # Test with cutoff = 2.5 (should connect atoms within 2.5 distance)
        converter_medium = GraphConverter(cutoff=2.5, pbc=False)
        data_medium = converter_medium.convert(pos=pos, cell=cell)
        assert data_medium.num_edges == 5 * 2  # 0-1, 1-2, 0-2, 1-3, 2-3 connections

        # Test with cutoff = 4.0 (should connect all atoms)
        converter_large = GraphConverter(cutoff=4.0, pbc=False)
        data_large = converter_large.convert(pos=pos, cell=cell)
        assert data_large.num_edges == 6 * 2  # All possible connections

    def test_edge_distance_calculation(self, simple_converter):
        """Test that edge distances are calculated correctly."""
        pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [3.0, 4.0, 0.0],  # Distance = 5.0
            ]
        )
        cell = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])

        data = simple_converter.convert(pos=pos, cell=cell)

        # With cutoff 2.5, no edges should be created
        assert data.num_edges == 0

        # Test with larger cutoff
        converter_large = GraphConverter(cutoff=6.0, pbc=False)
        data_large = converter_large.convert(pos=pos, cell=cell)

        # Should have 2 edges (A->B and B->A)
        assert data_large.num_edges == 2

        # Distance should be 5.0
        expected_distance = 5.0
        assert np.allclose(
            data_large.edge_dist,
            [expected_distance, expected_distance],
            atol=1e-6,
        )

    def test_empty_input(self, simple_converter):
        """Test behavior with empty input."""
        pos = np.empty((0, 3))
        cell = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])

        data = simple_converter.convert(pos=pos, cell=cell)

        assert data.num_nodes == 0
        assert data.num_edges == 0
        assert data.edge_index.shape[1] == 0

    def test_single_atom(self, simple_converter):
        """Test behavior with single atom."""
        pos = np.array([[0.0, 0.0, 0.0]])
        cell = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])

        data = simple_converter.convert(pos=pos, cell=cell)

        assert data.num_nodes == 1
        assert data.num_edges == 0  # No edges for single atom
        assert data.edge_index.shape[1] == 0

    def test_pbc_requires_cell(self):
        """Test that PBC converter requires cell parameters."""
        converter = GraphConverter(cutoff=5.0, pbc=True)
        pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        with pytest.raises(
            ValueError, match="Cell parameters are required when pbc is True"
        ):
            converter.convert(pos=pos)

    def test_pbc_with_cell(self, pbc_converter):
        """Test PBC converter with cell parameters."""
        pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )
        cell = np.array(
            [
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
            ]
        )

        data = pbc_converter.convert(pos=pos, cell=cell)

        assert isinstance(data, Data)
        assert data.num_nodes == 3
        assert data.num_edges > 0  # Should create some edges

    def test_convert_ase_atoms_method(self, simple_converter):
        """Test the convert_ase_atoms convenience method."""
        try:
            from ase import Atoms
            from ase.calculators.calculator import NullCalculator

            atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]], cell=[10, 10, 10])
            atoms.calc = NullCalculator()

            data = simple_converter.convert_ase_atoms(atoms)

            assert isinstance(data, Data)
            assert data.num_nodes == 2
            assert data.num_edges == 2  # H-H bond both directions
        except ImportError:
            pytest.skip("ASE not available")

    def test_convert_pymatgen_structure_method(self, simple_converter):
        """Test the convert_pymatgen_structure convenience method."""
        try:
            from pymatgen.core import Lattice, Structure

            lattice = Lattice.cubic(10.0)
            structure = Structure(lattice, ["H", "H"], [[0, 0, 0], [0.1, 0, 0]])

            data = simple_converter.convert_pymatgen_structure(structure)

            assert isinstance(data, Data)
            assert data.num_nodes == 2
            assert data.num_edges == 2  # H-H bond both directions
        except ImportError:
            pytest.skip("PyMatGen not available")


# Additional test functions for edge cases
def test_converter_initialization_parameters():
    """Test different initialization parameters."""
    # Test default values
    converter_default = GraphConverter()
    assert converter_default.cutoff == 5.0
    assert converter_default.pbc is True
    assert converter_default.three_body_cutoff == 3.0

    # Test custom values
    converter_custom = GraphConverter(cutoff=10.0, pbc=False, three_body_cutoff=8.0)
    assert converter_custom.cutoff == 10.0
    assert converter_custom.pbc is False
    assert converter_custom.three_body_cutoff == 8.0

    # Test None three_body_cutoff
    converter_no_three_body = GraphConverter(three_body_cutoff=None)
    assert converter_no_three_body.three_body_cutoff is None


def test_converter_repr():
    """Test string representation of the converter."""
    converter = GraphConverter(cutoff=3.0, pbc=False, three_body_cutoff=2.0)
    repr_str = repr(converter)

    assert "GraphConverter" in repr_str


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__])
