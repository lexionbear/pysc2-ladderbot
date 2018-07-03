## Inspired by  StevenBrown 
"""A base agent to write custom scripted agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.agents import base_agent
from pysc2.lib import actions, features, units

from enum import IntEnum, auto
import random

class ContextEnum(IntEnum):
  NumBases = auto()

  FeatureDimension = auto()

class MacroGlobalFeatureSet(IntEnum):
  MineralCount = auto()
  GasCount = auto()
  Population = auto()
  PopulationCap = auto() # essentially same infomation as overlord count
  FreeSupply = auto() # populationCap - population
  WorkerCount = auto()
  ArmyCount = auto()
  WorkerIdle = auto()
  LarvaCount = auto()

  BaseCount = auto()
  ExtractorCount = auto()
 
  WorkerOnMineral = auto()
  WorkerOnGas = auto()
  WorkerBuilding = auto()

  QueenCount = auto()


  FeatureDimension = auto()

class MacroPerBaseFeatureSet(IntEnum):
  IsMain = auto()
  MineralFieldCount = auto()
  WorkerOnMineral = auto()

  GasCount = auto()
  WorkerOnGas = auto()

  WorkerBuilding = auto()

  QueenCount = auto()
  IsInjected = auto()

  FeatureDimension = auto()


# Functions
_NOOP = actions.FUNCTIONS.no_op.id

class ZergMacroAgent(base_agent.BaseAgent):
  def __init__(self):
    super(ZergMacroAgent, self).__init__()

    #self.attack_coordinates = None
    self.unitSpace = [units.Zerg.Hatchery, units.Zerg.Drone, units.Zerg.Overlord, units.Zerg.SpawningPool, units.Zerg.Queen]

    # TODO: understand what is purified mineral
    self.neutralUnitSpace = [units.Neutral.MineralField, units.Neutral.MineralField750, units.Neutral.RichMineralField, units.Neutral.RichMineralField750]

    self.mineral = 0
    self.reward = 0

    self.startX = 0
    self.startY = 0

    self.globalActionQueue = []
  
  def can_do(self, obs, action):
    return action in obs.observation.available_actions

  def get_units_by_type(self, obs, unit_type):
    return [unit for unit in obs.observation.feature_units
            if unit.unit_type == unit_type]

  def moveDroneToMineral(self, obs):
    if(obs.observation.player.idle_worker_count >= 1):
      minerals = self.get_units_by_type(obs, units.Neutral.MineralField)
      if(len(minerals) > 0):
        actionUnit = []
        actionUnit.append((actions.FUNCTIONS.select_idle_worker(),actions.FUNCTIONS.select_idle_worker.id))

        mineral = random.choice(minerals)
        actionUnit.append((actions.FUNCTIONS.Harvest_Gather_screen("now", (mineral.x, mineral.y)),actions.FUNCTIONS.Harvest_Gather_screen.id))

        self.globalActionQueue.append(actionUnit)
        return True

    return False

  def buildDroneAction(self, obs):
    larvae = self.get_units_by_type(obs, units.Zerg.Larva)
    if len(larvae) > 0:
      larva = random.choice(larvae)

      actionUnit = []
      actionUnit.append((actions.FUNCTIONS.select_point("select_all_type", (larva.x,
                                                                larva.y)), actions.FUNCTIONS.select_point.id))

      actionUnit.append((actions.FUNCTIONS.Train_Drone_quick("now"), actions.FUNCTIONS.Train_Drone_quick.id))
    
      self.globalActionQueue.append(actionUnit)
      return True

    return False

  def buildOverlordAction(self, obs):
    larvae = self.get_units_by_type(obs, units.Zerg.Larva)
    if len(larvae) > 0:
      larva = random.choice(larvae)

      actionUnit = []
      actionUnit.append((actions.FUNCTIONS.select_point("select_all_type", (larva.x,
                                                                larva.y)), actions.FUNCTIONS.select_point.id))

      actionUnit.append((actions.FUNCTIONS.Train_Overlord_quick("now"), actions.FUNCTIONS.Train_Overlord_quick.id))
    
      self.globalActionQueue.append(actionUnit)
      return True

    return False

  def isSameActionUnit(self, actionUnit1, actionUnit2):
    if(len(actionUnit1) != len(actionUnit2)):
      return False

    actionUnitLen = len(actionUnit1)
    for i in range(actionUnitLen):
      action1 = actionUnit1[i]
      action2 = actionUnit2[i]

      if(action1[1] != action2[1]):
        # TODO: shall we check not just function id?
        return False

    return True


  def dedupConsecutiveSameOrder(self):
    if(len(self.globalActionQueue)>2):
      i = 0
      while(i in range(len(self.globalActionQueue) - 1)):
        curActionUnit = self.globalActionQueue[i]
        nextActionUnit = self.globalActionQueue[i+1]

        if(self.isSameActionUnit(curActionUnit, nextActionUnit)):
          del self.globalActionQueue[i]
        else:
          i+=1

    return self.globalActionQueue

  def executeDequeue(self, obs):
    if(len(self.globalActionQueue)>0):
      actionUnit = self.globalActionQueue.pop(0)

      if(len(actionUnit) == 0):
        return actions.FUNCTIONS.no_op()
      else:
        firstActionPair = actionUnit.pop(0)
        actionFunc = firstActionPair[0]
        actionFuncId = firstActionPair[1]

        if self.can_do(obs, actionFuncId):
          if(len(actionUnit) > 0):
            self.globalActionQueue = [actionUnit] + self.globalActionQueue

          return actionFunc

    return actions.FUNCTIONS.no_op()



  def step(self, obs):
    super(ZergMacroAgent, self).step(obs)

    # if obs.first():
    #   player_y, player_x = (obs.observation.feature_minimap.player_relative ==
    #                         features.PlayerRelative.SELF).nonzero()
    #   self.startX = player_x.mean()
    #   self.startY = player_y.mean()

    globalFeatures = self.globalMacroFeatureExtractor(obs)

    if(globalFeatures[MacroGlobalFeatureSet.FreeSupply] <= 1):
      self.buildOverlordAction(obs)
    else:
      self.buildDroneAction(obs)

    self.dedupConsecutiveSameOrder()
    return self.executeDequeue(obs)
    #return actions.FunctionCall(_NOOP, [])

  def globalMacroFeatureExtractor(self, obs):
    f = [0]*MacroGlobalFeatureSet.FeatureDimension

    f[MacroGlobalFeatureSet.MineralCount] = obs.observation.player.minerals
    f[MacroGlobalFeatureSet.GasCount] = obs.observation.player.vespene

    f[MacroGlobalFeatureSet.Population] = obs.observation.player.food_used
    f[MacroGlobalFeatureSet.PopulationCap] = obs.observation.player.food_cap
    f[MacroGlobalFeatureSet.FreeSupply] = (obs.observation.player.food_cap - obs.observation.player.food_used)

    f[MacroGlobalFeatureSet.WorkerCount] = obs.observation.player.food_workers
    f[MacroGlobalFeatureSet.WorkerIdle] = obs.observation.player.idle_worker_count
    f[MacroGlobalFeatureSet.ArmyCount] = obs.observation.player.army_count

    f[MacroGlobalFeatureSet.LarvaCount] = obs.observation.player.larva_count

    f[MacroGlobalFeatureSet.BaseCount] = len(self.get_units_by_type(obs, units.Zerg.Hatchery))
    
    return f

