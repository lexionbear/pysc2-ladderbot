import sys
import platform
from absl import logging
from absl import app
from absl import flags

from pysc2.env import sc2_env
from pysc2.env import run_loop

from pysc2.env import remote_sc2_env

# !!! LOAD YOUR BOT HERE !!!
from simple_agent import SimpleAgent
from zerg_agent_step7 import ZergAgent
from zerg_macro import ZergMacroAgent

RACE = sc2_env.Race.zerg

STEP_MUL = 4 #16 per second, 16/4 * 60 = 240 APM
AGENT_INTERFACE_FORMAT = sc2_env.parse_agent_interface_format(
    feature_screen=84,
    feature_minimap=64,
    rgb_screen=None,
    rgb_minimap=None,
    action_space="FEATURES", #FEATURES or RGB
    use_feature_units=True)

# Start game
def main(argv):
    logging.info("Starting local game...")

    maxSecond = 300
    max_steps = 16/STEP_MUL * 1.4 * maxSecond #adjusting for 1.4 for "faster" game speed setting
    max_episodes = 1

    players = []
    agents = []

    players.append(sc2_env.Agent(RACE))
    agents.append(ZergMacroAgent(max_steps))
    
    players.append(sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy))

    with sc2_env.SC2Env(
        map_name= "AbyssalReef", #"Simple64",
        players=players,
        visualize=False,
        step_mul=STEP_MUL,
        agent_interface_format=AGENT_INTERFACE_FORMAT) as env:

        # hack to override maxEpisodes as it doesn't seem to work
        # TODO: there is probably an unreported exception in the try catch at: https://github.com/deepmind/pysc2/blob/7b7afd7eeae985e6498855ac368c865ed9d527fb/pysc2/env/run_loop.py 
        total_episodes = 0
        while not max_episodes or total_episodes < max_episodes:
            total_episodes += 1
            run_loop.run_loop(agents, env, max_steps, 1)
            


if __name__ == '__main__':
    app.run(main)
