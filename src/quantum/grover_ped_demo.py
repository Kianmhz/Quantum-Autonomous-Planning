# src/quantum/grover_ped_demo.py
import json, math
from pathlib import Path

from src.planning.config import PlanConfig
from src.planning.candidates import make_accel_profiles
from src.planning.evaluator import eval_candidate

SNAP_PATH = Path("snapshots/ped_scenario_tick.json")

def main():
    snap = json.loads(SNAP_PATH.read_text())
    cfg = PlanConfig(
        dt=snap["cfg"]["dt"],
        horizon_s=snap["cfg"]["horizon_s"],
        v_ref=snap["cfg"]["v_ref"],
        d_safe=snap["cfg"]["d_safe"],
    )

    ego_loc = snap["ego"]["loc"]
    ego_vel = snap["ego"]["vel"]
    ped_loc = snap["ped"]["loc"]
    ped_vel = snap["ped"]["vel"]
    ego_half_width = snap["ego"]["half_width"]
    t_world = snap["sim_time"]

    # For offline demo we approximate lane as 1D straight line along ego velocity
    v0 = math.hypot(ego_vel[0], ego_vel[1])
    s0 = 0.0
    # Build a fake straight polyline
    N_pts = 400
    ds = 1.0
    heading = math.atan2(ego_vel[1], ego_vel[0]) if v0 > 0.1 else 0.0
    ux, uy = math.cos(heading), math.sin(heading)
    lane_points = []
    for i in range(N_pts):
        lane_points.append(
            type("P", (), {
                "x": ego_loc[0] + ux * i * ds,
                "y": ego_loc[1] + uy * i * ds,
                "z": ego_loc[2],
            })()
        )

    # Simple constant-velocity pedestrian predictor
    def ped_pred(t):
        dt = t - t_world
        if dt < 0.0: dt = 0.0
        return type("P", (), {
            "x": ped_loc[0] + ped_vel[0] * dt,
            "y": ped_loc[1] + ped_vel[1] * dt,
            "z": ped_loc[2],
        })()

    profiles = make_accel_profiles(v0, cfg)
    print(f"[Offline] Testing {len(profiles)} profiles at snapshot t={t_world:.2f}s")

    best_name, best_cost, best_diag = None, float("inf"), None
    for name, prof in profiles.items():
        valid, cost, diag = eval_candidate(
            prof, v0, s0, lane_points, ped_pred,
            cfg, t_world=t_world, ego_half_width=ego_half_width
        )
        print(f"  - {name:14s} valid={valid} cost={cost:.2f} "
              f"dmin={diag.get('clearance_min', 0):.2f}")
        if valid and cost < best_cost:
            best_name, best_cost, best_diag = name, cost, diag

    print(f"\n[Offline] Best classical profile: {best_name} "
          f"cost={best_cost:.2f} dmin={best_diag.get('clearance_min', 0):.2f}")

if __name__ == "__main__":
    main()
