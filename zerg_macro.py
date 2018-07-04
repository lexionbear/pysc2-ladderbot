from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.agents import base_agent
from pysc2.lib import actions, features, units

from enum import IntEnum, auto, Enum
import random

import heapq
import unit_build_time
import data_IO

EnableDebug = False

class AutoNumber(Enum):
  def __new__(cls):
    value = len(cls.__members__)  # note no + 1
    obj = object.__new__(cls)
    obj._value_ = value
    return obj

  def __int__(self):
    return self.value



class MacroGlobalFeatureSet(AutoNumber):
  MineralCount = ()
  GasCount = ()
  Population = ()
  PopulationCap = () # essentially same infomation as overlord count
  FreeSupply = () # populationCap - population
  WorkerCount = ()
  ArmyCount = ()
  WorkerIdle = ()
  LarvaCount = ()

  WorkerBuilding = ()
  PopulationBuilding = ()

  #QueenCount = ()
  #BaseCount = ()

  #ExtractorCount = ()
  #WorkerOnMineral = ()
  #WorkerOnGas = ()

  FeatureDimension = ()

class MacroPerBaseFeatureSet(AutoNumber):
  IsMain = ()
  MineralFieldCount = ()
  WorkerOnMineral = ()

  GasCount = ()
  WorkerOnGas = ()

  WorkerBuilding = ()

  QueenCount = ()
  IsInjected = ()

  FeatureDimension = ()

class MacroRewardRecord(AutoNumber):
  MineralCount = ()
  FeatureDimension = ()

class ActionRecord(AutoNumber):
  # record all actions
  # note it does not record all effective action, which is recorded separately
  BuildWorker = ()
  BuildOverloard = ()
  FeatureDimension = ()

UnitBuildingMap = {
  units.Zerg.Drone : MacroGlobalFeatureSet.WorkerBuilding,
  units.Zerg.Overlord : MacroGlobalFeatureSet.PopulationBuilding
}

class ZergMacroAgent(base_agent.BaseAgent):
  def __init__(self, MaxStep):
    super(ZergMacroAgent, self).__init__()
    self.pauseDebug = "0"

    self.maxStep = MaxStep
    self.reset()
  
  def reset(self):
    super(ZergMacroAgent, self).reset()
    self.mineral = 0
    self.reward = 0

    self.startX = 0
    self.startY = 0

    self.internalStepCounter = 0

    self.globalActionQueue = []
    self.buildHeapq = []

    self.globalFeatureRecord = []
    self.globalRewardRecord = []
    self.ActionRecord = []
    self.EffectiveRecord = []

    self.performedLastOperation = False

    return

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
        # Qustion: shall we check not just function id?
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
        return actions.FUNCTIONS.no_op(), actions.FUNCTIONS.no_op.id
      else:
        firstActionPair = actionUnit.pop(0)
        actionFunc = firstActionPair[0]
        actionFuncId = firstActionPair[1]

        if self.can_do(obs, actionFuncId):
          if(len(actionUnit) > 0):
            self.globalActionQueue = [actionUnit] + self.globalActionQueue

          # if it is a build action, register it in build queue
          if actionFuncId in unit_build_time.UnitBuildActionSet:
            unitId = unit_build_time.UnitBuildAction[actionFuncId]
            unitBuildTimeStep = unit_build_time.UnitBuildTime[unitId]
            expectFinishTime = self.internalStepCounter + unitBuildTimeStep
            heapq.heappush(self.buildHeapq, (expectFinishTime, unitId))

          return actionFunc, actionFuncId

    return actions.FUNCTIONS.no_op(), actions.FUNCTIONS.no_op.id

  # TODO: port in online learning
  def trainModel(self, buffer):
    return True

  def initStep(self, obs):
    player_y, player_x = (obs.observation.feature_minimap.player_relative ==
                            features.PlayerRelative.SELF).nonzero()
    self.startX = player_x.mean()
    self.startY = player_y.mean()

    self.reset()
    return

  def lastStep(self,obs):
    print('Final Mineral:', str(self.mineral))
    print('Final Step:', str(self.internalStepCounter))
    
    gf_path, _ = data_IO.genLatestFile(".//data", "globalFeatures")
    data_IO.export2DArray(self.globalFeatureRecord, gf_path)

    rw_path, _ = data_IO.genLatestFile(".//data", "macroRewards")
    data_IO.export2DArray(self.globalRewardRecord, rw_path)

    action_path, _ = data_IO.genLatestFile(".//data", "actions")
    data_IO.export2DArray(self.ActionRecord, action_path)

    effectiveAction_path, _ = data_IO.genLatestFile(".//data", "effectiveAction")
    data_IO.export2DArray(self.EffectiveRecord, effectiveAction_path)

    self.performedLastOperation = True
    
    return

  def step(self, obs):
    super(ZergMacroAgent, self).step(obs)

    if EnableDebug and (self.pauseDebug == "1" or self.internalStepCounter % 100 == 0):
      self.pauseDebug = input("pause prompt \n")
      print('current Step:', str(self.internalStepCounter))
      print('Units In Queue', str(len(self.buildHeapq)))

    if obs.first():
      self.initStep(obs)

    self.internalStepCounter += 1
    self.expireBuiltUnitFromQueue()
    
    globalFeatures = self.globalMacroFeatureExtractor(obs)
    rewards = self.MacroRewardExtractor(obs)
    
    self.globalFeatureRecord.append(globalFeatures)
    self.globalRewardRecord.append(rewards)
    
    # ------- Detect last iteration -----------
    if obs.last() or (self.internalStepCounter >= self.maxStep - 4 and self.maxStep > 0):
      # capture 1 second before the end
      if not self.performedLastOperation:
        self.lastStep(obs)

    # ------------------------------------------

    # ------- Core sub-Models -----------------------------
    actionVector = [0] * int(ActionRecord.FeatureDimension)

    if(globalFeatures[int(MacroGlobalFeatureSet.FreeSupply)] <= 1): 
      self.buildOverlordAction(obs)
      actionVector[int(ActionRecord.BuildOverloard)] = 1
    else:
      self.buildDroneAction(obs)
      actionVector[int(ActionRecord.BuildWorker)] = 1

    self.ActionRecord.append(actionVector)
    # -----------------------------------------------------
    # heuristic filters
    self.dedupConsecutiveSameOrder()

    # ------- Core final ranker ----------------------------


    actionFunc, actionFuncId = self.executeDequeue(obs) 
    self.EffectiveRecord.append(actionFuncId)
    return actionFunc


  def MacroRewardExtractor(self,obs):
    f = [0]*int(MacroRewardRecord.FeatureDimension)
    f[int(MacroRewardRecord.MineralCount)] = obs.observation.player.minerals

    return f

  def globalMacroFeatureExtractor(self, obs):
    f = [0]*int(MacroGlobalFeatureSet.FeatureDimension)

    self.mineral = obs.observation.player.minerals

    f[int(MacroGlobalFeatureSet.MineralCount)] = obs.observation.player.minerals
    f[int(MacroGlobalFeatureSet.GasCount)] = obs.observation.player.vespene

    f[int(MacroGlobalFeatureSet.Population)] = obs.observation.player.food_used
    f[int(MacroGlobalFeatureSet.PopulationCap)] = obs.observation.player.food_cap
    f[int(MacroGlobalFeatureSet.FreeSupply)] = (obs.observation.player.food_cap - obs.observation.player.food_used)

    f[int(MacroGlobalFeatureSet.WorkerCount)] = obs.observation.player.food_workers
    f[int(MacroGlobalFeatureSet.WorkerIdle)] = obs.observation.player.idle_worker_count
    f[int(MacroGlobalFeatureSet.ArmyCount)] = obs.observation.player.army_count

    f[int(MacroGlobalFeatureSet.LarvaCount)] = obs.observation.player.larva_count
    #f[int(MacroGlobalFeatureSet.BaseCount)] = len(self.get_units_by_type(obs, units.Zerg.Hatchery))

    self.extractUnitInBuild(f)
    return f

  def extractUnitInBuild(self, f):
    if len(self.buildHeapq) == 0:
      return

    for i in range(len(self.buildHeapq)):
      unitInBuildId = self.buildHeapq[i][1]

      # simulated switch statement
      # TODO: more optimized lookup out of stack
      featureId = int(UnitBuildingMap.get(unitInBuildId, -1))

      if featureId != -1:
        f[featureId] += 1

    return

  def expireBuiltUnitFromQueue(self):
    if len(self.buildHeapq) == 0:
      return

    finishTimeStep, unitId = heapq.heappop(self.buildHeapq)

    while finishTimeStep <= self.internalStepCounter:
      if(len(self.buildHeapq) == 0):
        return
      finishTimeStep, unitId = heapq.heappop(self.buildHeapq)

    if finishTimeStep > self.internalStepCounter:
      heapq.heappush(self.buildHeapq, (finishTimeStep, unitId))

    return

  