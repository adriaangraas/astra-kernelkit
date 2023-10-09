"""Performance of using multiple streams for small reconstructions."""
import itertools
import cupy as cp
import kernelkit as kk
import numpy as np
from tqdm import tqdm


def geometries(N_u, N_v, N_theta=1000):
    # custom geometry, 99x141 detector, 1mm pixel size
    # 10cm source-object distance, 20cm object-detector distance along x-axis
    geom_t0 = kk.ProjectionGeometry(
        source_position=[-10., 0., 0.],
        detector_position=[20., 0., 0.],
        detector=kk.Detector(
            rows=N_v, cols=N_u,
            pixel_height=.01, pixel_width=.01))
    geom_t0 = kk.rotate(geom_t0,
                        yaw=.5 * np.pi, roll=.2 * np.pi, pitch=.1 * np.pi)
    angles = np.linspace(0, 2 * np.pi, N_theta, False)
    return [kk.rotate(geom_t0, yaw=a) for a in angles]


NR_SAMPLES = 100
WITH_UPDATE = True
NR_WARMUPS = 100
TEXTURE_TYPE = 'array'
N_x, N_y, N_z, N_u, N_v, N_theta = 100, 100, 1, 1000, 1, 500

# setup
proj_geoms = geometries(N_u, N_v, N_theta)
vol_geom = kk.resolve_volume_geometry(
    shape=[N_x, N_y, N_z],
    extent_min=[-.1, None, None],
    extent_max=[.1, None, None])
gt = cp.random.rand(*vol_geom.shape, dtype=cp.float32)
projs = kk.fp(gt, proj_geoms, vol_geom)
vol = cp.zeros_like(gt)

# use a non-null stream
cp.cuda.Stream().use()

try:
    print(f"Without CUDA streams/graphs the speed is...")
    ptor = kk.BackProjector(texture_type=TEXTURE_TYPE)
    ptor.volume = vol
    ptor.volume_geometry = vol_geom
    ptor.projections = cp.asarray(projs, dtype=cp.float32)
    ptor.projection_geometry = proj_geoms
    ptor()
    for _ in tqdm(range(NR_SAMPLES)):
        if WITH_UPDATE:
            ptor.projections = projs
        ptor()
except KeyboardInterrupt:
    pass

#  using multiple num_streams with an own graph
for nr_streams in (1, 4,):
    try:
        streams = [cp.cuda.Stream(non_blocking=True) for _ in range(nr_streams)]
        graphs = []
        ptors = []
        for stream in streams:
            ptor = kk.BackProjector(texture_type=TEXTURE_TYPE)
            ptor.volume = vol
            ptor.volume_geometry = vol_geom
            ptor.projections = cp.asarray(projs, dtype=cp.float32)
            ptor.projection_geometry = proj_geoms
            for _ in range(NR_WARMUPS):
                if WITH_UPDATE:
                    ptor.projections = projs
                ptor()
            ptors.append(ptor)

            stream.use()
            stream.begin_capture()
            if WITH_UPDATE:
                ptor.projections = projs
            ptor()
            graph = stream.end_capture()
            graphs.append(graph)

        print(f"Using {nr_streams} streams with CUDA graphs the speed is...")
        ps = itertools.cycle(zip(ptors, streams, graphs))
        for _ in tqdm(range(NR_SAMPLES)):
            ptor, stream, graph = next(ps)
            with stream:
                graph.launch()
    except KeyboardInterrupt:
        pass