from .reckless_roaming_dancing_druid_v2 import RecklessRoamingDancingDruid

__all__ = [
    'RecklessRoamingDancingDruid',
    'POTENTIAL_CONTROLLERS'
]

r2d2_items_ranking = {
    'potion': 0, # 'potion' is not a weapon, but it is the most important item
    'axe': 1,
    'sword': 2,
    'amulet': 3,
    'knife': 4,
    'bow': 5,
    'bow_loaded': 5,
    'bow_unloaded': 5
}

POTENTIAL_CONTROLLERS = [
    RecklessRoamingDancingDruid('R2D2', r2d2_items_ranking)
]