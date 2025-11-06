from .config import PlanConfig
from .candidates import make_accel_profiles
from .evaluator import eval_candidate

def choose_accel_for_tick(v0, s0, lane_points, ped_pred, cfg: PlanConfig,
                          t_world: float = 0.0, ego_half_width: float = 1.0):
    
    profiles = make_accel_profiles(v0, cfg)
    best = (None, float("inf"), None, None)  # (name, cost, a0, diag)

    for name, prof in profiles.items():
        valid, cost, diag = eval_candidate(
            prof, v0, s0, lane_points, ped_pred,
            cfg, t_world=t_world, ego_half_width=ego_half_width
        )
        if valid and cost < best[1]:
            best = (name, cost, prof[0], diag)

    return best  # may be (None, inf, None, None)