# sample centerline ahead of ego
import math
import carla

from .geometry import fwd_vec

def build_lane_polyline(world, ego, max_m=200.0, ds=1.0):
    amap = world.get_map()
    wp = amap.get_waypoint(ego.get_location(), project_to_road=True,
                           lane_type=carla.LaneType.Driving)
    pts = []
    dist = 0.0
    while dist <= max_m and wp is not None:
        pts.append(wp.transform.location)
        nxt = wp.next(ds)
        if not nxt: break
        wp = nxt[0]
        dist += ds
    # s0 is our “index” along pts; we start at 0 each tick (local frame)
    v = ego.get_velocity()
    v0 = math.hypot(v.x, v.y)
    return pts, 0.0, v0

def follow_spectator(world, target, dist=7.5, height=2.5):
    spec = world.get_spectator()
    tf = target.get_transform()
    fwd = fwd_vec(tf.rotation)
    cam_loc = tf.location - fwd * dist
    cam_loc.z += height
    spec.set_transform(carla.Transform(cam_loc, tf.rotation))