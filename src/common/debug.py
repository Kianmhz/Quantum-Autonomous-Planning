# --- Utility ---
import carla

def label(world, loc, text, color=carla.Color(0,255,0), life=10.0):
    world.debug.draw_string(loc, text, False, color, life, True)