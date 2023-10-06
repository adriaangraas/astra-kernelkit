"""Demonstration of using a PyTorch CUDAGraph with KernelKit operators."""
import cupy as cp
import cupyx as cpx
import kernelkit as kk
from kernelkit import BackProjector, ForwardProjector
from kernelkit.torch_support import AutogradOperator
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class SimpleModel(nn.Module):
    """Convolutional layers after backprojection"""
    INTERMEDIATE_CHANNELS = 10

    def __init__(self, bp):
        super().__init__()
        self.bp = bp
        layers = [
            nn.Conv3d(C, self.INTERMEDIATE_CHANNELS, kernel_size=3, padding=1,
                      bias=False), nn.ReLU(),
            nn.Conv3d(self.INTERMEDIATE_CHANNELS, self.INTERMEDIATE_CHANNELS,
                      kernel_size=3, padding=1, bias=False), nn.ReLU(),
            nn.Conv3d(self.INTERMEDIATE_CHANNELS, self.INTERMEDIATE_CHANNELS,
                      kernel_size=3, padding=1, bias=False), nn.ReLU(),
            nn.Conv3d(self.INTERMEDIATE_CHANNELS, self.INTERMEDIATE_CHANNELS,
                      kernel_size=3, padding=1, bias=False), nn.ReLU(),
            nn.Conv3d(self.INTERMEDIATE_CHANNELS, self.INTERMEDIATE_CHANNELS,
                      kernel_size=3, padding=1, bias=False), nn.ReLU(),
            nn.Conv3d(self.INTERMEDIATE_CHANNELS, self.INTERMEDIATE_CHANNELS,
                      kernel_size=3, padding=1, bias=False), nn.ReLU(),
            nn.Conv3d(self.INTERMEDIATE_CHANNELS, C, kernel_size=3, padding=1,
                      bias=False)]
        self.block = nn.Sequential(*layers)

    def forward(self, y):
        # backproject the projections
        x = AutogradOperator.apply(y, self.bp)
        # then pipe through a small CNN
        return self.block(x)


if __name__ == '__main__':
    N = 20
    Z = 5
    VOL_SHAPE = (N, N, Z)
    DETECTOR_SHAPE = (2*Z, N)
    NR_ANGLES = 3
    B, C = 32, 1

    # create projection geometries
    geom_t0 = kk.ProjectionGeometry(
        source_position=[-10., 0., 0.],
        detector_position=[10., 0., 0.],
        detector=kk.Detector(*DETECTOR_SHAPE, .01, .01))
    angles = np.linspace(0.0, 2*np.pi, NR_ANGLES, False)
    geoms = [kk.rotate(geom_t0, yaw=a) for a in angles]

    # create volume geometry
    vol_geom = kk.resolve_volume_geometry(
        shape=VOL_SHAPE,
        extent_min=(-1.0, None, None),
        extent_max=(1.0, None, None))

    # create a ProjectorOperator
    ftor = ForwardProjector(volume_axes=(2, 1, 0), )
    ftor.projection_geometry = geoms
    ftor.volume_geometry = vol_geom

    btor = BackProjector(
        voxels_per_block=(16, 32, 6 if VOL_SHAPE[-1] > 1 else 1),
        texture_type='array',
        use_graph=False,
        volume_axes=(2, 1, 0),)
    btor.projection_geometry = geoms
    btor.volume_geometry = vol_geom

    op = kk.ProjectorOperator(
        projector=ftor,
        backprojector=btor)

    # create a Torch stream, and set CuPy to use it
    s = torch.cuda.Stream()
    gcs = cp.cuda.ExternalStream(s.cuda_stream)
    gcs.use()

    # training set-up
    model = SimpleModel(op.T).cuda()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # modified from: https://pytorch.org/docs/stable/cuda.html#cuda-graphs
    # -------------------------------------------------------------------------
    NITERS = 100
    static_input = torch.randn(B, C, NR_ANGLES, *DETECTOR_SHAPE, device='cuda')
    static_target = torch.randn(B, C, *reversed(VOL_SHAPE), device='cuda')

    # warmup
    # Uses static_input and static_target here for convenience,
    # but in a real setting, because the warmup includes optimizer.step()
    # you must use a few batches of real data.
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in tqdm(range(NITERS)):
            optimizer.zero_grad(set_to_none=True)
            y_pred = model(static_input)
            loss = loss_fn(y_pred, static_target)
            loss.backward()
            optimizer.step()
    torch.cuda.current_stream().wait_stream(s)

    # capture
    g = torch.cuda.CUDAGraph()
    # Sets grads to None before capture, so backward() will create
    # .grad attributes with allocations from the graph's private pool
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.graph(g, stream=s) as graph:
        static_y_pred = model(static_input)
        static_loss = loss_fn(static_y_pred, static_target)
        static_loss.backward()
        optimizer.step()

    real_inputs = [torch.rand_like(static_input) for _ in range(1)]
    real_targets = [torch.rand_like(static_target) for _ in range(1)]
    for data, target in zip(real_inputs, real_targets):
        # Fills the graph's input memory with new data to compute on
        for i in tqdm(range(NITERS)):
            static_input.copy_(data)
            static_target.copy_(target)
        # replay() includes forward, backward, and step.
        # You don't even need to call optimizer.zero_grad() between iterations
        # because the captured backward refills static .grad tensors in place.
        for i in tqdm(range(NITERS)):
            g.replay()
    # end of taken section from the PyTorch docs
    # -------------------------------------------------------------------------

    # make a groundtruth volume, in page locked memory
    gt = cpx.zeros_pinned(VOL_SHAPE, dtype=cp.float32)
    gt[90:110, 90:110].fill(1.)
    y = op(cp.asarray(gt.transpose(2, 1, 0)))

    plt.figure(0)
    for i in range(NR_ANGLES):
        plt.cla()
        plt.imshow(y[i].get())
        plt.pause(1.)
        plt.show()

    x = op.T(y)

    plt.figure(1)
    plt.imshow(x[0, :, :].get())
    plt.pause(1.)
    plt.show()