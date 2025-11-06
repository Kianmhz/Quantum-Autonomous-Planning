# --- Geometry ---
import math
import carla

def fwd_vec(rot):
    yaw = math.radians(rot.yaw)
    return carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0)

def right_vec(rot):
    yaw = math.radians(rot.yaw + 90.0)
    return carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0)

def move_behind(tf, distance):
    fwd = fwd_vec(tf.rotation)
    new_tf = carla.Transform(tf.location - fwd * distance, tf.rotation)
    return new_tf

def yaw_left_unit(rot):
    yaw = math.radians(rot.yaw)
    return math.sin(yaw), -math.cos(yaw)

# --- Map transforms ---
def transform_on_other_side(world, base_tf, side="left",
                            lanes_guess=3, median_guess=2.0,
                            forward=8.0, up=0.5):
    amap = world.get_map()
    base_wp = amap.get_waypoint(base_tf.location, project_to_road=True,
                                lane_type=carla.LaneType.Driving)

    lane_w = base_wp.lane_width or 3.5
    jump = lanes_guess * lane_w + median_guess
    offset = +jump if side == "left" else -jump

    ahead_wp = base_wp.next(forward)[0]
    ahead_tf = ahead_wp.transform
    lx, ly = yaw_left_unit(ahead_tf.rotation)
    cand = carla.Location(
        x=ahead_tf.location.x + lx * offset,
        y=ahead_tf.location.y + ly * offset,
        z=ahead_tf.location.z
    )

    target_wp = amap.get_waypoint(cand, project_to_road=True,
                                  lane_type=carla.LaneType.Driving)
    target_wp = target_wp.next(4.0)[0]
    tf = target_wp.transform
    tf.location.z += up
    return tf