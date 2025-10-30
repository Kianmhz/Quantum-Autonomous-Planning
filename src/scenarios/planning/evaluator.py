from .config import PlanConfig
from .candidates import index_to_point, rollout_1d
import math

def min_clearance(points, ss, ped_pred, cfg: PlanConfig):
    t_list = [i*cfg.dt for i in range(len(ss))]
    dmin = 1e9
    for s_i, t in zip(ss, t_list):
        e = index_to_point(points, s_i)
        p = ped_pred(t)
        dx, dy = e.x - p.x, e.y - p.y
        d = math.hypot(dx, dy)
        if d < dmin: dmin = d
    return dmin

def eval_candidate(a_profile, v0, s0, lane_points, ped_pred, cfg: PlanConfig):
    vs, ss = rollout_1d(v0, s0, a_profile, cfg)
    clr = min_clearance(lane_points, ss, ped_pred, cfg)
    # hard constraint
    if clr < cfg.d_safe:
        return False, 1e9, {"clearance_min": clr}
    # soft cost: speed tracking + accel smoothness
    v_err = sum((v - cfg.v_ref)**2 for v in vs)
    a_cost = sum(a*a for a in a_profile)
    j_cost = sum((a2-a1)**2 for a1,a2 in zip(a_profile[:-1], a_profile[1:]))
    J = 5*v_err + 2*a_cost + 1*j_cost
    return True, J, {"clearance_min": clr, "v_end": vs[-1]}