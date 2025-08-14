import pytest
import torch
from torch_geometric.data import Data

from m3gnet.model.layers import RadiusGraphLayer


class TestRadiusGraphLayer:
    """Test suite for RadiusGraphLayer."""

    @pytest.fixture
    def simple_layer(self):
        """Create a simple RadiusGraphLayer instance."""
        return RadiusGraphLayer(cutoff=2.5)

    @pytest.fixture
    def large_cutoff_layer(self):
        """Create a layer with larger cutoff for more edges."""
        return RadiusGraphLayer(cutoff=6.0)

    # @pytest.fixture
    # def three_body_layer(self):
    #     """Create a layer with three-body interactions enabled."""
    #     return RadiusGraphLayer(
    #         cutoff=3.0,
    #         enable_three_body=True,
    #         three_body_cutoff=2.0,
    #     )

    def test_simple_unbatched_graph(self, simple_layer):
        """
        Tests graph creation for a single, simple structure (no batching).
        """
        # 1. Arrange
        # Points: A=(0,0,0), B=(2,0,0), C=(5,0,0)
        pos = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ]
        )

        # 2. Act
        data = simple_layer.forward(pos)

        # 3. Assert
        # We expect an edge between A and B, and B and A (undirected)
        # The distance is sqrt((2-0)^2) = 2.0, which is < 2.5
        # The distance between A and C is 5.0, which is > 2.5

        assert isinstance(data, Data)
        assert data.num_nodes == 3
        assert data.num_edges == 2

        # Check that edge distances are correct
        expected_distances = [2.0, 2.0]  # A-B and B-A distances
        assert torch.allclose(
            data.edge_attr, torch.tensor(expected_distances), atol=1e-6
        )

        # Check that positions are preserved
        assert torch.allclose(data.pos, pos)

    def test_no_edges_with_small_cutoff(self):
        """Test that no edges are created when cutoff is too small."""
        # Points: A=(0,0,0), B=(3,0,0), C=(6,0,0)
        pos = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
            ]
        )

        layer = RadiusGraphLayer(cutoff=1.0)  # Very small cutoff
        data = layer.forward(pos)

        assert data.num_nodes == 3
        assert data.num_edges == 0
        assert data.edge_index.shape[1] == 0

    def test_all_edges_with_large_cutoff(self, large_cutoff_layer):
        """Test that all possible edges are created with large cutoff."""
        # Points: A=(0,0,0), B=(1,0,0), C=(2,0,0)
        pos = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )

        data = large_cutoff_layer.forward(pos)

        # With cutoff 6.0, all atoms should be connected
        # Expected: 6 edges (A-B, B-A, A-C, C-A, B-C, C-B)
        assert data.num_nodes == 3
        assert data.num_edges == 6

        # Sort edge attributes to ensure consistent ordering
        sorted_edge_attr = torch.sort(data.edge_attr)[0]

        # Check that all distances are correct
        expected_distances = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0]
        assert torch.allclose(
            sorted_edge_attr, torch.tensor(expected_distances), atol=1e-6
        )

    def test_batched_graphs(self, simple_layer):
        """Test graph creation with batched structures."""
        # Two structures: [A,B] and [C,D]
        pos = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # Structure 0, atom 0 -- A
                [1.0, 0.0, 0.0],  # Structure 0, atom 1 -- B
                [10.0, 0.0, 0.0],  # Structure 1, atom 0 -- C
                [11.0, 0.0, 0.0],  # Structure 1, atom 1 -- D
                [12.0, 0.0, 0.0],  # Structure 1, atom 2 -- E
            ]
        )
        batch = torch.tensor([0, 0, 1, 1, 1])

        data = simple_layer.forward(pos, batch)

        assert data.num_nodes == 5
        # Should have edges within each structure but not between structures
        assert data.num_edges == 8  # A-B, B-A, C-D, D-C, C-E, E-C, D-E, E-D

        # Check that edge distances are correct
        expected_distances = [1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0]
        assert torch.allclose(
            data.edge_attr, torch.tensor(expected_distances), atol=1e-6
        )

    def test_3d_coordinates(self, simple_layer):
        """Test graph creation with 3D coordinates."""
        # Points in 3D space
        pos = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [3.0, 3.0, 3.0],
            ]
        )

        data = simple_layer.forward(pos)

        # Distance A-B = sqrt(1² + 1² + 1²) = sqrt(3) ≈ 1.73 < 2.5
        # Distance A-C = sqrt(3² + 3² + 3²) = sqrt(27) ≈ 5.20 > 2.5
        # Distance B-C = sqrt(2² + 2² + 2²) = sqrt(12) ≈ 3.46 > 2.5

        assert data.num_nodes == 3
        assert data.num_edges == 2  # Only A-B connection

        # Check that the edge distance is approximately sqrt(3)
        expected_distance = torch.sqrt(torch.tensor(3.0))
        assert torch.allclose(
            data.edge_attr,
            torch.tensor([expected_distance, expected_distance]),
            atol=1e-6,
        )

    def test_edge_index_format(self, simple_layer):
        """Test that edge_index has correct format."""
        pos = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )

        data = simple_layer.forward(pos)

        # Should have 2 edges (A->B and B->A)
        assert data.edge_index.shape == (2, 2)
        assert data.edge_index.dtype == torch.long

        # Check that edge_index contains valid node indices
        assert torch.all(data.edge_index >= 0)
        assert torch.all(data.edge_index < data.num_nodes)

    def test_edge_attr_format(self, simple_layer):
        """Test that edge_attr has correct format."""
        pos = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )

        data = simple_layer.forward(pos)

        # Should have 2 edge attributes (distances)
        assert data.edge_attr.shape == (2,)
        assert data.edge_attr.dtype == torch.float32

        # All distances should be positive
        assert torch.all(data.edge_attr > 0)

    # def test_three_body_layer_creation(self, three_body_layer):
    #     """Test that three-body layer is created correctly."""
    #     assert three_body_layer.cutoff == 3.0
    #     assert three_body_layer.enable_three_body is True
    #     assert three_body_layer.three_body_cutoff == 2.0

    # def test_three_body_layer_forward(self, three_body_layer):
    #     """Test three-body layer forward pass (currently unimplemented)."""
    #     pos = torch.tensor([
    #         [0.0, 0.0, 0.0],
    #         [1.0, 0.0, 0.0],
    #         [2.0, 0.0, 0.0],
    #     ])

    #     # Should not raise an error even though three-body is not implemented
    #     data = three_body_layer.forward(pos)

    #     assert isinstance(data, Data)
    #     assert data.num_nodes == 3
    #     # Should still create edges based on cutoff
    #     assert data.num_edges > 0

    def test_different_cutoff_values(self):
        """Test layer behavior with different cutoff values."""
        pos = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        )

        # Test with cutoff = 1.5 (should only connect adjacent atoms)
        layer_small = RadiusGraphLayer(cutoff=1.5)
        data_small = layer_small.forward(pos)
        assert data_small.num_edges == 3 * 2  # Only 0-1, 1-2, 2-3 connections

        # Test with cutoff = 2.5 (should connect atoms within 2.5 distance)
        layer_medium = RadiusGraphLayer(cutoff=2.5)
        data_medium = layer_medium.forward(pos)
        assert data_medium.num_edges == 5 * 2  # 0-1, 1-2, 0-2, 1-3, 2-3 connections

        # Test with cutoff = 4.0 (should connect all atoms)
        layer_large = RadiusGraphLayer(cutoff=4.0)
        data_large = layer_large.forward(pos)
        assert data_large.num_edges == 6 * 2  # All possible connections

    def test_edge_distance_calculation(self, simple_layer):
        """Test that edge distances are calculated correctly."""
        pos = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [3.0, 4.0, 0.0],  # Distance = 5.0
            ]
        )

        data = simple_layer.forward(pos)

        # With cutoff 2.5, no edges should be created
        assert data.num_edges == 0

        # Test with larger cutoff
        layer_large = RadiusGraphLayer(cutoff=6.0)
        data_large = layer_large.forward(pos)

        # Should have 2 edges (A->B and B->A)
        assert data_large.num_edges == 2

        # Distance should be 5.0
        expected_distance = 5.0
        assert torch.allclose(
            data_large.edge_attr,
            torch.tensor([expected_distance, expected_distance]),
            atol=1e-6,
        )

    def test_empty_input(self, simple_layer):
        """Test behavior with empty input."""
        pos = torch.empty(0, 3)

        data = simple_layer.forward(pos)

        assert data.num_nodes == 0
        assert data.num_edges == 0
        assert data.edge_index.shape[1] == 0

    def test_single_atom(self, simple_layer):
        """Test behavior with single atom."""
        pos = torch.tensor([[0.0, 0.0, 0.0]])

        data = simple_layer.forward(pos)

        assert data.num_nodes == 1
        assert data.num_edges == 0  # No edges for single atom
        assert data.edge_index.shape[1] == 0

    def test_device_consistency(self, simple_layer):
        """Test that output tensors are on the same device as input."""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=device)

            data = simple_layer.forward(pos)

            assert data.pos.device == device
            assert data.edge_index.device == device
            assert data.edge_attr.device == device

    def test_gradient_flow(self, simple_layer):
        """Test that gradients can flow through the layer."""
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], requires_grad=True)

        data = simple_layer.forward(pos)

        # The output should maintain requires_grad
        assert data.pos.requires_grad

        # Test backward pass
        loss = data.pos.sum()
        loss.backward()

        # Check that gradients are computed
        assert pos.grad is not None
        assert pos.grad.shape == pos.shape


# Additional test functions for edge cases
def test_layer_initialization_parameters():
    """Test different initialization parameters."""
    # Test default values
    layer_default = RadiusGraphLayer()
    assert layer_default.cutoff == 5.0
    assert layer_default.enable_three_body is False
    assert layer_default.three_body_cutoff == 4.0

    # Test custom values
    layer_custom = RadiusGraphLayer(
        cutoff=10.0, enable_three_body=True, three_body_cutoff=8.0
    )
    assert layer_custom.cutoff == 10.0
    assert layer_custom.enable_three_body is True
    assert layer_custom.three_body_cutoff == 8.0


def test_layer_repr():
    """Test string representation of the layer."""
    layer = RadiusGraphLayer(cutoff=3.0, enable_three_body=True)
    repr_str = repr(layer)

    assert "RadiusGraphLayer" in repr_str
    assert "cutoff=3.0" in repr_str
    assert "enable_three_body=True" in repr_str


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__])
