from dataclasses import dataclass

@dataclass
class PlanConfig:
    dt: float = 0.1          # how often we simulate (1 tick = 0.1s)
    horizon_s: float = 3.0   # how far ahead we plan (3s)
    v_ref: float = 5.0       # desired speed
    d_safe: float = 2.0      # min allowed distance to pedestrian
    a_min: float = -4.0      # hard braking limit
    a_max: float =  2.0      # max accel
    jerk_max: float = 6.0    # optional comfort bound