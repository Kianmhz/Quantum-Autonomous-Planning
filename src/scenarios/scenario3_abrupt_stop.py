# src/scenarios/scenario3_abrupt_stop.py
import time, math
import carla

from .helpers import label, follow_spectator

# --- Scenario Parameters ---
TOWN = "Town03"
SIM_DT = 0.01                 
RUNTIME_S = 20.0

EGO_SPEED_MS = 2         # desired cruise speed
EGO_CRUISE_THROTTLE = 0.6     # simple cruise throttle ceiling

LEAD_INIT_GAP_M = 45.0        # lead starts this far ahead (same lane)
DELTA_TRIG_S    = 3.0         # at Δ, lead begins braking hard
LEAD_BRAKE_SECS = 2.0
# -----------------

def set_sync(world, enabled=True, dt=SIM_DT):
    s = world.get_settings()
    s.synchronous_mode = enabled
    s.fixed_delta_seconds = dt if enabled else None
    s.substepping = False
    world.apply_settings(s)

def choose_straight_spawn(world):
    spawns = world.get_map().get_spawn_points()
    return spawns[170]

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    # Load/ensure map
    world = client.get_world()
    if world.get_map().name.split('/')[-1] != TOWN:
        world = client.load_world(TOWN)
        time.sleep(0.5)

    bp = world.get_blueprint_library()
    ego_bp  = (bp.filter('vehicle.tesla.model3') or bp.filter('vehicle.*'))[0]
    lead_bp = (bp.filter('vehicle.audi.tt')      or bp.filter('vehicle.*'))[0]

    set_sync(world, True, SIM_DT)
    actors = []

    try:
        # ---------- Spawn EGO ----------
        base_sp = choose_straight_spawn(world)

        ego_tf = base_sp

        label(world, base_sp.location, "EGO_BASE", carla.Color(0,255,0))
        label(world, ego_tf.location, "EGO_SPAWN", carla.Color(0,200,255))

        ego = world.spawn_actor(ego_bp, ego_tf)
        actors.append(ego)
        ego.set_autopilot(False)

        # ---------- Spawn LEAD ahead on same lane using waypoints ----------
        amap = world.get_map()
        ego_wp = amap.get_waypoint(ego_tf.location, project_to_road=True,
                                   lane_type=carla.LaneType.Driving)
        # advance along lane by LEAD_INIT_GAP_M
        ahead_wp = ego_wp.next(LEAD_INIT_GAP_M)[0]
        lead_tf  = ahead_wp.transform
        lead_tf.location.z += 0.5  # small lift to avoid clipping
        lead = world.try_spawn_actor(lead_bp, lead_tf)

        actors.append(lead)
        lead.set_autopilot(False)

        label(world, lead_tf.location, "LEAD_START", carla.Color(255,140,0))

        # ---------- Loop timing ----------
        total_ticks = int(RUNTIME_S / SIM_DT)
        trig_tick   = int(DELTA_TRIG_S / SIM_DT)
        brake_ticks = int(LEAD_BRAKE_SECS / SIM_DT)

        print("[Scenario 3] Abrupt Stop: Lead will hard-brake at Δ = %.1fs." % DELTA_TRIG_S)

        # ---------- Main simulation loop ----------
        for tick in range(total_ticks):
            world.tick()
            follow_spectator(world, ego)

            # EGO: very simple cruise near target
            v = ego.get_velocity()
            speed = math.hypot(v.x, v.y)
            ego.apply_control(carla.VehicleControl(
                throttle = EGO_CRUISE_THROTTLE if speed < EGO_SPEED_MS else 0.0,
                brake    = 0.3 if speed > EGO_SPEED_MS + 0.5 else 0.0,
                steer    = 0.0
            ))

            # LEAD: pace just under ego speed until trigger
            if tick < trig_tick:
                v_lead = lead.get_velocity()
                s_lead = math.hypot(v_lead.x, v_lead.y)
                target = EGO_SPEED_MS * 0.95
                lead.apply_control(carla.VehicleControl(
                    throttle = 0.5 if s_lead < target else 0.0,
                    brake    = 0.3 if s_lead > target + 0.5 else 0.0,
                    steer    = 0.0
                ))

            # At Δ: hard braking window
            if trig_tick <= tick < trig_tick + brake_ticks:
                if tick == trig_tick:
                    print("[Scenario 3] Lead begins hard braking.")
                lead.apply_control(carla.VehicleControl(throttle=0.0, brake=0.9, hand_brake=False))

            # After window: hold full brake (stopped)
            if tick >= trig_tick + brake_ticks:
                lead.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

            # Optional safety: early stop if ego gets too close
            delta = lead.get_location() - ego.get_location()
            gap_m = math.hypot(delta.x, delta.y)
            if gap_m < 5.0:
                print("[Scenario 3] Early stop: ego within 5 m of lead.")
                break

        print("[Scenario 3] Finished.")

    finally:
        for a in actors[::-1]:
            try:
                if hasattr(a, "stop"): a.stop()
            except Exception:
                pass
            try:
                a.destroy()
            except Exception:
                pass
        set_sync(world, False)
        print("[Scenario 3] Clean shutdown; sync disabled.")

if __name__ == "__main__":
    main()
