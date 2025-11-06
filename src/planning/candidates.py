from .config import PlanConfig

# --- Candidates (accelerations) ---
def make_accel_profiles(v0, cfg: PlanConfig):
    N = int(cfg.horizon_s / cfg.dt)
    def const(a): return [a]*N
    profiles = {
        "keep": const(0.0),
        "comfort_brake": const(-2.0),
        "hard_brake": const(-4.0),
    }
    if v0 > 0.5:
        profiles["creep"] = const(-1.0)
    return profiles

# project 1D along sampled polyline: index-as-distance (1 idx ≈ 1m)
def rollout_1d(v0, s0, a_profile, cfg: PlanConfig):
    v, s = v0, s0
    vs, ss = [v0], [s0]
    for a in a_profile:
        a = max(cfg.a_min, min(cfg.a_max, a))
        v = max(0.0, v + a * cfg.dt)
        s = s + v * cfg.dt      # ds ≈ v*dt; our polyline is ~1m per index
        vs.append(v); ss.append(s)
    return vs, ss

def index_to_point(points, s_idx):
    j = max(0, min(len(points)-1, int(round(s_idx))))
    return points[j]
