from utils import *

@sar_test
def test_patch_dgl():
    import dgl
    original_gat_edge_softmax = dgl.nn.pytorch.conv.gatconv.edge_softmax
    original_dotgat_edge_softmax = dgl.nn.pytorch.conv.dotgatconv.edge_softmax
    original_agnn_edge_softmax = dgl.nn.pytorch.conv.agnnconv.edge_softmax

    import sar
    sar.patch_dgl()

    assert original_gat_edge_softmax == dgl.nn.functional.edge_softmax
    assert original_dotgat_edge_softmax == dgl.nn.functional.edge_softmax
    assert original_agnn_edge_softmax == dgl.nn.functional.edge_softmax

    assert dgl.nn.pytorch.conv.gatconv.edge_softmax == sar.patched_edge_softmax
    assert dgl.nn.pytorch.conv.dotgatconv.edge_softmax == sar.patched_edge_softmax
    assert dgl.nn.pytorch.conv.agnnconv.edge_softmax == sar.patched_edge_softmax