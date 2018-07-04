from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.agents import base_agent
from pysc2.lib import actions, features, units


Step_Mul = 4

UnitBuildActionSet = set([actions.FUNCTIONS.Train_Drone_quick.id, actions.FUNCTIONS.Train_Overlord_quick.id])

UnitBuildAction = {
    actions.FUNCTIONS.Train_Drone_quick.id : units.Zerg.Drone,
    actions.FUNCTIONS.Train_Overlord_quick.id : units.Zerg.Overlord
}


# Reference : http://us.battle.net/sc2/en/game/unit/overlord
UnitBuildTime = {
    units.Zerg.Drone : 17 * 16 / Step_Mul, #* 1.4
    units.Zerg.Overlord : 25 * 16 / Step_Mul,
    units.Zerg.Queen : 50 * 16 / Step_Mul
}