from .config import PlanConfig
from .candidates import index_to_point, rollout_1d
import math

def min_clearance(points, ss, ped_pred, cfg, t_world,
                  ego_half_width_m=1.0, ped_radius_m=0.35):
    dmin = 1e9
    for k, s_i in enumerate(ss):
        e = index_to_point(points, s_i)
        t = t_world + k * cfg.dt
        p = ped_pred(t)
        d = math.hypot(e.x - p.x, e.y - p.y)
        d_eff = d - (ego_half_width_m + ped_radius_m)
        if d_eff < dmin:
            dmin = d_eff
    return dmin


def eval_candidate(a_profile, v0, s0, lane_points, ped_pred,
                   cfg: PlanConfig, t_world=0.0, ego_half_width=1.0):
    vs, ss = rollout_1d(v0, s0, a_profile, cfg)

    dmin = min_clearance(lane_points, ss, ped_pred, cfg,
                         t_world, ego_half_width_m=ego_half_width,
                         ped_radius_m=0.35)

    if dmin < cfg.d_safe:
        return False, 1e9, {"clearance_min": dmin}

    v_err = sum((v - cfg.v_ref)**2 for v in vs)
    a_cost = sum(a*a for a in a_profile)
    j_cost = sum((a2 - a1)**2 for a1, a2 in zip(a_profile[:-1], a_profile[1:]))
    prox_penalty = sum(max(0, (cfg.d_safe + 2.0 - dmin))**2 for d in [dmin])
    J = 5*v_err + 2*a_cost + 1*j_cost + 10*prox_penalty

    return True, J, {"clearance_min": dmin, "v_end": vs[-1]}
