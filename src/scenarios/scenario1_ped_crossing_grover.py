# src/scenarios/scenario1_ped_crossing.py
import time, random
import carla

from src.planning.config import PlanConfig
from src.planning.prediction import ped_predictor, is_ped_relevant

from src.common.geometry import fwd_vec, right_vec, move_behind, transform_on_other_side
from src.common.world import build_lane_polyline, follow_spectator
from src.common.control import accel_to_controls

from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

from src.planning.candidates import make_accel_profiles
from src.planning.evaluator import eval_candidate
from src.quantum.grover_ped_demo import (
    grover_oracle_from_costs,
    grover_diffusion,
    ACTION_TO_BIT,
    BIT_TO_ACTION,
)

# ------- Scenario Parameters -------
TOWN = "Town03"          # urban map
EGO_SPEED_MS = 9         # ≈ 32 km/h realistic city driving
CROSS_DELAY_S = 4      # pedestrian starts crossing after 2 seconds
PED_SPEED_MS = 2.5       # ≈ 9 km/h brisk walking
AHEAD_M = 43.0           # keep this (avoids object collision)
LATERAL_M = 8.0          # pedestrian crossing offset
SIM_DT = 0.01            # simulation time step
RUNTIME_S = 12.0         # total scenario duration
PLANNING_DT = 0.20       # plan at 5 Hz
PLAN_EVERY = max(1, int(PLANNING_DT / SIM_DT))
SMOOTH_ALPHA = 0.4       # smooth control transitions
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

        # Step 2: jump to the other tunnel side
        ego_tf = transform_on_other_side(
            world, base_sp, side="left", lanes_guess=3,
            median_guess=2.0, forward=8.0, up=0.5
        )
        ego_tf = move_behind(ego_tf, 23.0)

        # Step 3: spawn ego
        ego = world.spawn_actor(ego_bp, ego_tf)
        actors.append(ego)
        ego.set_autopilot(False)
        ego_half_width = ego.bounding_box.extent.y

        # Step 4: compute pedestrian spawn relative to new ego
        fwd = fwd_vec(ego_tf.rotation)
        right = right_vec(ego_tf.rotation)
        ped_loc = ego_tf.location + fwd * AHEAD_M + right * LATERAL_M
        ped_tf = carla.Transform(ped_loc, ego_tf.rotation)
        ped = world.spawn_actor(walker_bp, ped_tf)
        actors.append(ped)

        # --- Initialize prediction and control state ---
        sim_time = 0.0
        ped_start_time = float("inf")
        ped_pred = ped_predictor(
            ped_loc, carla.Vector3D(-right.x, -right.y, 0.0),
            PED_SPEED_MS, start_time=ped_start_time
        )

        cfg = PlanConfig(dt=PLANNING_DT, horizon_s=3.0,
                         v_ref=EGO_SPEED_MS, d_safe=3.5)

        target_thr, target_brk = 0.0, 0.0
        apply_thr, apply_brk = 0.0, 0.0

        ticks_to_delay = int(CROSS_DELAY_S / SIM_DT)
        ticks_total = int(RUNTIME_S / SIM_DT)
        print("[Scenario] Starting. Pedestrian crosses after delay.")
        # ---------------------------------------------------
        #  Simulation loop
        # ---------------------------------------------------
        for tick in range(ticks_total):
            world.tick()
            sim_time += SIM_DT
            follow_spectator(world, ego)

            # Replan at lower frequency
            if tick % PLAN_EVERY == 0:
                lane_points, s0, v0 = build_lane_polyline(
                    world, ego, max_m=200.0, ds=1.0
                )
                if is_ped_relevant(ego, ped.get_location(), max_range=60.0):
                    # --- Generate candidate profiles ---
                    all_profiles = make_accel_profiles(v0, cfg)
                    profiles = {n: p for n, p in all_profiles.items() if n in ACTION_TO_BIT}

                    per_action = {}
                    for pname, prof in profiles.items():
                        valid, cst, d = eval_candidate(
                            prof, v0, s0, lane_points, ped_pred, cfg,
                            t_world=sim_time, ego_half_width=ego_half_width,
                        )
                        if valid:
                            per_action[pname] = cst

                    if not per_action:
                        print(f"[Grover@{sim_time:.2f}s] No valid profiles, keeping previous accel.")
                        a0 = 0.0
                    else:
                        # --- Run Grover search ---
                        qc = QuantumCircuit(2, 2)
                        qc.h([0, 1])
                        oracle = grover_oracle_from_costs(per_action)
                        diffusion = grover_diffusion(2)
                        qc.compose(oracle, [0, 1], inplace=True)
                        qc.compose(diffusion, [0, 1], inplace=True)
                        qc.measure([0, 1], [0, 1])

                        backend = Aer.get_backend("aer_simulator")
                        result = backend.run(transpile(qc, backend), shots=128).result()
                        counts = result.get_counts()

                        best_bits = max(counts, key=counts.get)
                        best_action = BIT_TO_ACTION[best_bits]

                        print(f"[Grover@{sim_time:.2f}s] Action={best_action} ({counts})")

                        # --- Translate chosen action to acceleration ---
                        if best_action == "keep":
                            a0 = 0.0
                        elif best_action == "comfort_brake":
                            a0 = -2.0
                        elif best_action == "hard_brake":
                            a0 = -4.0
                        elif best_action == "creep":
                            a0 = -1.0
                        else:
                            a0 = 0.0

                    target_thr, target_brk = accel_to_controls(a0, v0, cfg.v_ref)

            # Smooth control transitions every tick
            def lerp(a, b, t): return a + t * (b - a)
            apply_thr = lerp(apply_thr, target_thr, SMOOTH_ALPHA)
            apply_brk = lerp(apply_brk, target_brk, SMOOTH_ALPHA)

            ego.apply_control(carla.VehicleControl(
                throttle=apply_thr, brake=apply_brk, steer=0.0
            ))

            # Trigger pedestrian crossing
            if tick == ticks_to_delay:
                ped_ctrl = carla.WalkerControl(
                    direction=carla.Vector3D(-right.x, -right.y, 0.0),
                    speed=PED_SPEED_MS
                )
                ped.apply_control(ped_ctrl)
                ped_start_time = sim_time
                ped_pred = ped_predictor(
                    ped.get_location(),
                    carla.Vector3D(-right.x, -right.y, 0.0),
                    PED_SPEED_MS, start_time=ped_start_time
                )
                print(f"[Scenario] Pedestrian started crossing at t={sim_time:.2f}s.")

            # Stop pedestrian after ~20 m
            if tick > ticks_to_delay:
                rel = ped.get_location() - ped_loc
                proj = rel.x * (-right.x) + rel.y * (-right.y)
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
