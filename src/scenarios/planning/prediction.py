import carla

# constant-velocity pedestrian prediction
def ped_predictor(ped_loc0, ped_dir_unit, ped_speed):
    def pred(t: float):
        return carla.Location(
            x=ped_loc0.x + ped_dir_unit.x * ped_speed * t,
            y=ped_loc0.y + ped_dir_unit.y * ped_speed * t,
            z=ped_loc0.z
        )
    return pred

# quick relevance check to start planning
def is_ped_relevant(ego, ped_loc0, max_range=60.0):
    e = ego.get_location()
    dx, dy = ped_loc0.x - e.x, ped_loc0.y - e.y
    return (dx*dx + dy*dy) ** 0.5 <= max_range