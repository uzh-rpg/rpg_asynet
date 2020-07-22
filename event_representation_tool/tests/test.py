import event_representations as er 
import numpy as np
import matplotlib.pyplot as plt


def virtual_events(optical_flow, N, circle_radius, starting_point, resolution):
    """
    optical flow: 2 numbers, vx, vy denoting the flow in x and y direction.
    N: Number of events
    Example:
        events = optical_flow([5, 6], 10000)
    """
    vx, vy = optical_flow

    # sample t and angle from uniform distribution
    time = np.sort(np.random.random((N,))).reshape((N,1))
    angle = np.random.random((N,1)) * 2*np.pi
    polarity = 2*(np.random.random((N,1)) > .5)-1

    # compute coordinates
    u0, v0 = starting_point
    x = (u0 + time * vx + np.cos(angle) * circle_radius).astype(np.int64)
    y = (v0 + time * vy + np.sin(angle) * circle_radius).astype(np.int64)

    # compute mask for events that are within image
    H, W = resolution
    mask = (x[:,0] >= 0) & (y[:,0] >= 0) & (x[:,0] < W) & (y[:,0] < H)

    events = np.concatenate([x[mask, :], y[mask, :], time[mask, :], polarity[mask, :]], 1)

    return events

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time

    np.random.seed(2)

    H = 50
    W = 50
    K = 15

    flow = 10
    radius = 5
    N_events = 2000
    start = 20
    horizon = 0  # Not implemented

    # generate fake events
    events1 = virtual_events([flow,0], N_events, radius, [start, start], [W, H])
    events2 = virtual_events([-flow,0], N_events, radius, [start+flow, start], [W, H])
    events2[:,2]+=1
    events = np.concatenate([events1,events2],0)

    # generate a queue of events
    tensor = er.event_queue_tensor(events, K, H, W, horizon)  # 2 x K x H x W

    fig, ax = plt.subplots(ncols=1)
    ax[0].imshow(tensor[0,:,20,:])
    plt.show()

