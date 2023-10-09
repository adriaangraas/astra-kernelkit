"""Demonstration of using a PyTorch CUDAGraph with KernelKit operators."""
import cupy as cp
import kernelkit as kk
from kernelkit import BackProjector, ForwardProjector
from kernelkit.torch_support import AutogradOperator
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class BackprojectorModel(nn.Module):
    """Convolutional layers after backprojection"""

    def __init__(self, proj_geoms, vol_geom):
        super().__init__()
        ftor = ForwardProjector()
        ftor.projection_geometry = proj_geoms
        ftor.volume_geometry = vol_geom
        btor = BackProjector(
            voxels_per_block=(16, 32, 6 if vol_geom.shape[-1] > 1 else 1))
        btor.projection_geometry = proj_geoms
        btor.volume_geometry = vol_geom
        self.bp = kk.ProjectorOperator(projector=ftor, backprojector=btor).T

    def forward(self, y):
        return AutogradOperator.apply(y, self.bp)


def run():
    N = 200
    Z = 10
    VOL_SHAPE = (N, N, Z)
    DETECTOR_SHAPE = (2*Z, N)
    NR_ANGLES = 5
    B = 32

    # create projection geometries
    geom_t0 = kk.ProjectionGeometry(
        source_position=[-10., 0., 0.],
        detector_position=[10., 0., 0.],
        detector=kk.Detector(*DETECTOR_SHAPE, .01, .01))
    angles = np.linspace(0.0, 2*np.pi, NR_ANGLES, False)
    proj_geoms = [kk.rotate(geom_t0, yaw=a) for a in angles]

    # create volume geometry
    vol_geom = kk.resolve_volume_geometry(
        shape=VOL_SHAPE,
        extent_min=(-1.0, None, None),
        extent_max=(1.0, None, None))

    # create a Torch stream, and set CuPy to use it
    s = torch.cuda.Stream()
    gcs = cp.cuda.ExternalStream(s.cuda_stream)
    gcs.use()

    # See: https://pytorch.org/docs/stable/cuda.html#cuda-graphs
    # -------------------------------------------------------------------------
    NITERS = 10
    static_y = torch.ones(B, 1, NR_ANGLES, *DETECTOR_SHAPE, device='cuda')
    static_x = torch.zeros(B, 1, *VOL_SHAPE, device='cuda')

    # training set-up
    model = BackprojectorModel(proj_geoms, vol_geom).cuda()

    # warmup
    # Uses static_input and static_target here for convenience,
    # but in a real setting, because the warmup includes optimizer.step()
    # you must use a few batches of real data.
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in tqdm(range(NITERS)):
            static_x[...] = model(static_y)
    torch.cuda.current_stream().wait_stream(s)

    plt.figure()
    plt.title("Usual backprojection.""")
    plt.imshow(static_x[0, 0, ..., 0].cpu().numpy())
    plt.show()

    # capture
    static_x.fill_(-1.)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):  # passing stream is important for CuPy
        static_x[...] = model(static_y)

    plt.figure()
    plt.title("Should remain all -1! Graph capture doesn't execute kernels.")
    plt.imshow(static_x[0, 0, ..., 0].cpu().numpy())
    plt.show()

    static_y.copy_(torch.randn_like(static_y))
    g.replay()

    plt.figure()
    plt.title("Should be backprojected randoms!")
    plt.imshow(static_x[0, 0, ..., 0].cpu().numpy())
    plt.show()


if __name__ == '__main__':
    run()
