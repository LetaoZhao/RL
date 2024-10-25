from functools import cached_property

import numpy as np
from pyboy.utils import WindowEvent
import math
import time

from pyboy_environment.environments.pokemon.pokemon_environment import (
    PokemonEnvironment,
)
from pyboy_environment.environments.pokemon import pokemon_constants as pkc


class PokemonBrock(PokemonEnvironment):
    def __init__(
        self,
        act_freq: int,
        emulation_speed: int = 0,
        headless: bool = False,
    ) -> None:

        valid_actions: list[WindowEvent] = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            # WindowEvent.PRESS_BUTTON_START,
        ]

        release_button: list[WindowEvent] = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            # WindowEvent.RELEASE_BUTTON_START,
        ]

        self.mapSwitch_count = 0
        # self.isMapReach = [0,0,0]

        super().__init__(
            act_freq=act_freq,
            task="brock",
            init_name="has_pokedex.state",
            emulation_speed=emulation_speed,
            valid_actions=valid_actions,
            release_button=release_button,
            headless=headless,
        )



    def _get_state(self) -> np.ndarray:
        # Implement your state retrieval logic here
        game_stats = self._generate_game_stats()

        return_state = self.get_used_state(game_stats)
        # print(return_state)

        # time.sleep(0.5)

        # if (self.count < 1000):
        #     self.count += 1
        # else:
        #     self.count = 0
        #     self.isMapReach = [0,0,0]

        return return_state
    


    def get_used_state(self,full_state):
        
        used_states = [
            full_state["location"]["x"],
            full_state["location"]["y"],
            full_state["location"]["map_id"]
        ]

        return used_states



    def _calculate_reward(self, new_state: dict) -> float:
        # Implement your reward calculation logic here

        return_score = 0.0

        # new_location_x = new_state["location"]["x"]
        # new_location_y = new_state["location"]["y"]
        # pre_location_x = self.prior_game_stats["location"]["x"]
        # pre_location_y = self.prior_game_stats["location"]["y"]

        if(self.mapSwitch_count != 0):
            if(self.mapSwitch_count < 2):
                self.mapSwitch_count += 1
            else:
                self.mapSwitch_count = 0


        if((new_state["location"]["map_id"] == 0) and (self.prior_game_stats["location"]["map_id"] == 40)):
            return_score += 1000
            self.mapSwitch_count = 1
        if((new_state["location"]["map_id"] == 40) and (self.prior_game_stats["location"]["map_id"] == 0)):
            return_score -= 1000
            self.mapSwitch_count = 1

        if ((new_state["location"]["map_id"] == 40) and (self.mapSwitch_count == 0)):
            new_dis = self.distance_to_target(new_state,[5,10])
            pre_dis = self.distance_to_target(self.prior_game_stats,[5,10])

            return_score += self.distance_potential_score(new_dis,pre_dis,10)

        if ((new_state["location"]["map_id"] == 0) and (self.mapSwitch_count == 0)):
            new_dis = self.distance_to_target(new_state,[9,14])
            pre_dis = self.distance_to_target(self.prior_game_stats,[9,14])
            # print("------------------")
            # print([new_dis,pre_dis])
            
            # return_score += self.distance_potential_score(new_dis,pre_dis,10)
            # print(return_score)
            return_score += self.distance_discrete_score(new_dis,pre_dis,2,200)
            # print(return_score)
            # print("------------------")
        
        # return_score += self.not_move_penalty(new_state,self.prior_game_stats,2)
        # print(return_score)
            

        return new_state["badges"] - self.prior_game_stats["badges"]



    def distance_to_target(self,state,target):

        x = state["location"]["x"]
        y = state["location"]["y"]

        distance = math.sqrt(math.pow((target[0]-x),2) + math.pow((target[1]-y),2))

        return distance
    
    
    def distance_potential_score(self,new_distance,pre_distance,gain):

        return_score = 0.0

        diff_distance = pre_distance - new_distance
        return_score = gain*diff_distance

        return return_score
    
    def distance_discrete_score(self,new_distance,pre_distance,step,gain):

        return_score = 0.0

        new_step = new_distance//step
        pre_step = pre_distance//step

        diff_step = pre_step - new_step
        return_score = gain*diff_step

        return return_score
    
    def not_move_penalty(self,new_state,pre_state,gain):
        x_move = new_state["location"]["x"] - pre_state["location"]["x"]
        y_move = new_state["location"]["y"] - pre_state["location"]["y"]

        if ((x_move == 0) and (y_move == 0)):
            return -gain
        else:
            return 0.0



    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if agent beats first gym (temporary)
        return game_stats["badges"] > self.prior_game_stats["badges"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        # Implement your truncation check logic here

        # Maybe if we run out of pokeballs...? or a max step count
        return self.steps >= 1000
