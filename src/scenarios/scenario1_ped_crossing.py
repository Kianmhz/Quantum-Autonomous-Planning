# src/scenarios/scenario1_ped_crossing.py
import time, random, math
import carla
from .helpers import fwd_vec, right_vec, move_behind, transform_on_other_side, label, follow_spectator

# ------- Scenario Parameters -------
TOWN = "Town03"          # urban map
EGO_SPEED_MS = 12        # desired speed (m/s)
CROSS_DELAY_S = 1.0      # Δ when pedestrian steps off curb
PED_SPEED_MS = 3.0        # walking speed
AHEAD_M = 43.0           # ped spawn ahead of ego
LATERAL_M = 10.0         # to the right (curb)
SIM_DT = 0.01            # simulation time step
RUNTIME_S = 10.0
# -----------------------------------

def set_sync(world, enabled=True, dt=SIM_DT):
    s = world.get_settings()
    s.synchronous_mode = enabled
    s.fixed_delta_seconds = dt if enabled else None
    s.substepping = False
    world.apply_settings(s)

def choose_straight_spawn(world):
    spawns = world.get_map().get_spawn_points()
    index = 141 
    return spawns[index]

# ---------------------------------------------------------------
#  Main Scenario
# ---------------------------------------------------------------

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    # Load map
    world = client.load_world(TOWN)
    time.sleep(0.5)

    bp = world.get_blueprint_library()
    ego_bp = (bp.filter('vehicle.tesla.model3') or bp.filter('vehicle.*'))[0]
    walker_bp = random.choice(bp.filter('walker.pedestrian.*'))

    # Reproducible
    set_sync(world, True, SIM_DT)

    actors = []
    try:
        # Step 1: pick a base spawn
        base_sp = choose_straight_spawn(world)

        # Step 2: jump to the other tunnel side (change side="right" if needed)
        ego_tf = transform_on_other_side(world, base_sp,
                                         side="left",
                                         lanes_guess=3,
                                         median_guess=2.0,
                                         forward=8.0,
                                         up=0.5)
        
        ego_tf = move_behind(ego_tf, 23.0)

        label(world, base_sp.location, "BASE", carla.Color(0,255,0))
        label(world, ego_tf.location, "OTHER SIDE", carla.Color(0,200,255))

        # Step 3: spawn ego
        ego = world.spawn_actor(ego_bp, ego_tf)
        actors.append(ego)
        ego.set_autopilot(False)

        # Step 4: compute pedestrian spawn relative to new ego
        fwd  = fwd_vec(ego_tf.rotation)
        right = right_vec(ego_tf.rotation)
        ped_loc = ego_tf.location + fwd * AHEAD_M + right * LATERAL_M
        ped_tf  = carla.Transform(ped_loc, ego_tf.rotation)
        ped = world.spawn_actor(walker_bp, ped_tf)
        actors.append(ped)

        world.debug.draw_string(ped_loc, "PED START", False,
                                carla.Color(255,0,0), 10.0, True)

        # Simulation loop
        ticks_to_delay = int(CROSS_DELAY_S / SIM_DT)
        ticks_total = int(RUNTIME_S / SIM_DT)
        print("[Scenario] Starting. Pedestrian crosses after delay.")

        for tick in range(ticks_total):
            world.tick()
            follow_spectator(world, ego)

            # Maintain target speed
            v = ego.get_velocity()
            speed = math.hypot(v.x, v.y)
            throttle = 1 if speed < EGO_SPEED_MS else 0.0
            brake = 0.2 if speed > EGO_SPEED_MS + 0.5 else 0.0
            ego.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=0.0))

            # Trigger pedestrian crossing
            if tick == ticks_to_delay:
                dir_vec = carla.Vector3D(-right.x, -right.y, -right.z)
                ped_ctrl = carla.WalkerControl(direction=dir_vec, speed=PED_SPEED_MS)
                ped.apply_control(ped_ctrl)
                print("[Scenario] Pedestrian started crossing.")

            # Stop pedestrian after ~20 m
            if tick > ticks_to_delay:
                rel = ped.get_location() - ped_loc
                proj = rel.x * (-right.x) + rel.y * (-right.y)  # calculate displacement
                if proj > 20.0:
                    ped.apply_control(carla.WalkerControl(speed=0.0))

        print("[Scenario] Finished.")

    finally:
        for a in actors[::-1]:
            try:
                if hasattr(a, "stop"):
                    a.stop()
            except Exception:
                pass
            try:
                a.destroy()
            except Exception:
                pass
        set_sync(world, False)
        print("[Scenario] Clean shutdown & sync disabled.")

if __name__ == "__main__":
    main()