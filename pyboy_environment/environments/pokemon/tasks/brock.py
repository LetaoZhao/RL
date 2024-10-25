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
        # self.location_potential_table = [
        #     []
        # ]

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

        time.sleep(0.1)

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

        # if(self.mapSwitch_count != 0):
        #     if(self.mapSwitch_count < 2):
        #         self.mapSwitch_count += 1
        #     else:
        #         self.mapSwitch_count = 0


        # if((new_state["location"]["map_id"] == 0) and (self.prior_game_stats["location"]["map_id"] == 40)):
        #     return_score += 1000
        #     self.mapSwitch_count = 1
        # if((new_state["location"]["map_id"] == 40) and (self.prior_game_stats["location"]["map_id"] == 0)):
        #     return_score -= 1000
        #     self.mapSwitch_count = 1

        # if ((new_state["location"]["map_id"] == 40) and (self.mapSwitch_count == 0)):
        #     new_dis = self.distance_to_target(new_state,[5,10])
        #     pre_dis = self.distance_to_target(self.prior_game_stats,[5,10])

        #     # return_score += self.distance_potential_score(new_dis,pre_dis,10)
        #     return_score += self.distance_discrete_score(new_dis,pre_dis,2,200)

        # if ((new_state["location"]["map_id"] == 0) and (self.mapSwitch_count == 0)):
        #     new_dis = self.distance_to_target(new_state,[9,7])
        #     pre_dis = self.distance_to_target(self.prior_game_stats,[9,7])
        #     # print("------------------")
        #     # print([new_dis,pre_dis])
            
        #     return_score += self.distance_potential_score(new_dis,pre_dis,10)
        #     # print(return_score)
        #     return_score += self.distance_discrete_score(new_dis,pre_dis,2,200)
        #     # print(return_score)
        #     # print("------------------")
        
        # # return_score += self.not_move_penalty(new_state,self.prior_game_stats,2)
        # # print(return_score)

        return_score = self.get_gride_potential_score(new_state,self.prior_game_stats)
        print(return_score)
            

        return return_score



    def get_gride_potential_score(self,new_state,pre_state):
        return_score = 0.0

        new_location = [new_state["location"]["x"],new_state["location"]["y"]]
        new_map = new_state["location"]["map_id"]
        pre_location= [pre_state["location"]["x"],pre_state["location"]["y"]]
        pre_map = pre_state["location"]["map_id"]

        target_location = self.get_target_location(new_map)

        if (new_map != pre_map):
            self.mapSwitch_count = 1
            return_score = self.get_map_change_score(new_map,pre_map)

        elif (self.mapSwitch_count != 0):
            if (self.mapSwitch_count < 2):
                self.mapSwitch_count += 1
            else:
                self.mapSwitch_count = 0
            return_score = 0.0

        else:
            new_step_to_target = self.get_step_to_target(new_location,target_location)
            pre_step_to_target = self.get_step_to_target(pre_location,target_location)

            diff_potential = self.get_potential_change(new_step_to_target,pre_step_to_target,new_map)
            return_score = self.get_potential_score(diff_potential,new_map)

        return return_score
    
    def get_target_location(self,map_id):
        location = [0,0]

        if (map_id == 40):
            location = [5,10]
        elif (map_id == 0):
            location = [9,4]
        else:
            location = [0,0]

        return location
    
    def get_map_change_score(self,new_map_id,pre_map_id):
        score = 0.0

        if ((new_map_id == 0) and (pre_map_id == 40)):
            score = 100
        elif ((new_map_id == 40) and (pre_map_id == 0)):
            score = -100
        else:
            score = 0.0

        return score
    
    def get_step_to_target(self,current_pos,target_pos):
        steps = 0

        x_step = abs(target_pos[0] - current_pos[0])
        y_step = abs(target_pos[1] - current_pos[1])
        steps = x_step + y_step

        return steps
    
    def get_potential_change(self,new_step,pre_step,map_id):
        new_potiential = 0
        pre_potiential = 0
        diff_potiential = 0

        if (map_id == 40):
            min_potiential = 0
            max_potiential = 6
        elif (map_id == 0):
            min_potiential = 7
            max_potiential = 23
        else:
            return diff_potiential
        
        max_step = max_potiential - min_potiential
        if (new_step > max_step):
            new_potiential = 0
        else:
            new_potiential = max_potiential - new_step

        if (pre_step > max_step):
            pre_potiential = 0
        else:
            pre_potiential = max_potiential - pre_step

        diff_potiential = new_potiential - pre_potiential

        return diff_potiential
    
    def get_potential_score(self,diff_potential,map_id):
        score = 0.0
        gain = 0

        if (map_id == 40):
            gain = 1
        elif (map_id == 0):
            gain = 2
        else:
            gain = 0
        
        score = gain*diff_potential

        return score

        

        






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
