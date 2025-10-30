def accel_to_controls(a0, v0, v_ref):
    # crude mapping; refine later
    if a0 is None:
        return 0.0, 1.0  # emergency brake
    if a0 >= 0:
        throttle = max(0.0, min(1.0, 0.2 + 0.15*a0 + 0.25*(v_ref - v0)))
        return throttle, 0.0
    else:
        brake = max(0.2, min(1.0, -a0/4.0))
        return 0.0, brake
