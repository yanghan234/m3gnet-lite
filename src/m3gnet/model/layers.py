import torch
from torch import nn
from torch_cluster import radius_graph
from torch_geometric.data import Data


class RadiusGraphLayer(nn.Module):
    """
    Convert a set of atomic coordinates and cell parameters into a PyG graph.

    Args:
        cutoff: Cutoff radius for the graph construction. Default is 5.0.
        enable_three_body: Whether to include three-body interactions. Default is False.
        three_body_cutoff: Cutoff radius for three-body interactions. Default is 4.0.
        device: Device to use for the graph construction. Default is "cpu".
    """

    def __init__(
        self,
        cutoff: float = 5.0,
        enable_three_body: bool = False,
        three_body_cutoff: float = 4.0,
    ):
        super().__init__()

        self.cutoff = cutoff
        self.enable_three_body = enable_three_body
        self.three_body_cutoff = three_body_cutoff

    def forward(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> Data:
        """
        Convert a set of atomic coordinates and cell parameters into a PyG graph.
        """
        edge_index, edge_dist = self._get_edge_distances(pos, batch)
        if self.enable_three_body:
            # TODO: Implement three-body interactions
            pass
        return Data(
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_dist,
        )

    def _get_edge_distances(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Get the distances between all pairs of atoms in the graph.
        """
        if batch is None:
            batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)

        edge_index = radius_graph(
            pos,
            r=self.cutoff,
            batch=batch,
        )
        edge_dist = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
        return edge_index, edge_dist

    def __repr__(self) -> str:
        return f"""
        RadiusGraphLayer(
            cutoff={self.cutoff},
            enable_three_body={self.enable_three_body},
            three_body_cutoff={self.three_body_cutoff}
        )
        """

    def __str__(self) -> str:
        return self.__repr__()
