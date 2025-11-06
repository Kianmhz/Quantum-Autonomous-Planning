import carla

# constant-velocity pedestrian prediction
def ped_predictor(ped_loc0, dir_unit, speed, start_time=0.0):
    def pred(t_world: float):
        if t_world < start_time:
            return ped_loc0
        dt = t_world - start_time
        return carla.Location(
            ped_loc0.x + dir_unit.x * speed * dt,
            ped_loc0.y + dir_unit.y * speed * dt,
            ped_loc0.z
        )
    return pred

# quick relevance check to start planning
def is_ped_relevant(ego, ped_loc0, max_range=60.0):
    e = ego.get_location()
    dx, dy = ped_loc0.x - e.x, ped_loc0.y - e.y
    return (dx*dx + dy*dy) ** 0.5 <= max_range