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
            WindowEvent.RELEASE_ARROW_DOWN,  # 0
            WindowEvent.RELEASE_ARROW_LEFT,  # 1
            WindowEvent.RELEASE_ARROW_RIGHT, # 2
            WindowEvent.RELEASE_ARROW_UP,    # 3
            WindowEvent.RELEASE_BUTTON_A,    # 4
            WindowEvent.RELEASE_BUTTON_B,    # 5
            # WindowEvent.RELEASE_BUTTON_START,
        ]

        self.pathOverX_count = 0
        self.pathOverY_count = 0
        self.mapSwitch_count = 0
        self.mapSwitch_count1 = 0
        self.notmove = 0
        self.step_action = 0

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
        # print(game_stats["map"])

        # time.sleep(0.1)

        return return_state
    
    def get_used_state(self,full_state):
        
        used_states = [
            full_state["location"]["x"],
            full_state["location"]["y"],
            full_state["location"]["map_id"],
            self.step_action
        ]

        return used_states



    def _calculate_reward(self, new_state: dict) -> float:
        # Implement your reward calculation logic here

        return_score = 0.0

        if(self.mapSwitch_count1 != 0):
            if(self.mapSwitch_count1 < 5):
                self.mapSwitch_count1 += 1
            else:
                self.mapSwitch_count1 = 0
                # print("end_switch")


        if((new_state["location"]["map_id"] == 0) and (self.prior_game_stats["location"]["map_id"] == 40)):
            return_score += 1000
            self.mapSwitch_count1 = 1
            # print("start_switch")
        if((new_state["location"]["map_id"] == 40) and (self.prior_game_stats["location"]["map_id"] == 0)):
            return_score -= 1000
            self.mapSwitch_count1 = 1
            # print("start_switch")

        if(self.mapSwitch_count1 == 0):
            # print("normal")
            return_score += self.distance_reward(new_state)
            # return_score += self.step_penalty(10)
            # return_score += self.collision_penalty(new_state)
            return_score += self.inMap_step_reward(new_state)
            return_score += self.not_move_penalty(new_state,self.prior_game_stats,2)
            return_score += self.notOK_action_penalty()
        else:
            # print("on_switch")
            return_score += 1
        
        # print(return_score)


        if (new_state["location"]["map_id"] == 12):
            return_score += 10000
            print("YEEEEEEEEEES")
            
        return return_score
    




    # def get_table_reward(self,new_state):
    #     score = 0.0

    #     map_id = self.prior_game_stats["location"]["map_id"]
    #     new_location = [new_state["location"]["x"],new_state["location"]["y"]]
    #     pre_location = [self.prior_game_stats["location"]["x"],self.prior_game_stats["location"]["y"]]
    #     action = self.step_action

    #     if (map_id == 40):
    #         if (action == 0):
    #             if(new_location[1] > pre_location[1]):
    #                 score += 10
    #             else:
    #                 score -= 10
    #         else:
    #             score -= 10

    #     elif (map_id == 0):
    #         if()
    




    def step_penalty(self,gain):
        return -gain
    
    def collision_penalty(self,new_state):
        score = 0

        new_location = [new_state["location"]["x"],new_state["location"]["y"]]
        pre_location = [self.prior_game_stats["location"]["x"],self.prior_game_stats["location"]["y"]]

        if(new_location == pre_location):
            if((self.step_action == 1) or (self.step_action == 2) or (self.step_action == 3) or (self.step_action == 0)):
                score -= 10

        return score
                        
    def inMap_step_reward(self,new_state):
        score = 0.0

        map_id = new_state["location"]["map_id"]
        new_location = [new_state["location"]["x"],new_state["location"]["y"]]
        pre_location = [self.prior_game_stats["location"]["x"],self.prior_game_stats["location"]["y"]]

        if (map_id == 0):
            if ((new_location[0] == 8) and (pre_location[0] != 8)):
                score += 1000
            if ((new_location[0] != 8) and (pre_location[0] == 8)):
                score -= 1000
            if ((new_location[1] <= 6) and (pre_location[1] > 6)):
                score += 1000
            if ((new_location[1] > 6) and (pre_location[1] <= 6)):
                score -= 1000

        return score
    
    
    def notOK_action_penalty(self):
        score = 0.0

        map_id = self.prior_game_stats["location"]["map_id"]

        if (map_id == 40):
            if ((self.step_action == 1) or (self.step_action == 2) or (self.step_action == 3) or (self.step_action == 4) or (self.step_action == 5)):
                score -= 1
        elif (map_id == 0):
            if ((self.step_action == 4) or (self.step_action == 5)):
                score -= 1

        return score

    def distance_reward(self,new_state):
        score = 0.0

        new_location = [new_state["location"]["x"],new_state["location"]["y"]]
        pre_location = [self.prior_game_stats["location"]["x"],self.prior_game_stats["location"]["y"]]
        new_map_id = new_state["location"]["map_id"]
        pre_map_id = self.prior_game_stats["location"]["map_id"]

        target = [0,0]
        new_diff = [0,0]
        pre_diff = [0,0]

        if (new_map_id == pre_map_id):
            if (new_map_id == 40):
                target = [5,12]
            elif (new_map_id == 0): 
                target = [8,0]
            else:
                target = [0,0]

            new_diff = [(target[0]-new_location[0]),(target[1]-new_location[1])]
            pre_diff = [(target[0]-pre_location[0]),(target[1]-pre_location[1])]

            if (np.sign(new_diff[0]) == np.sign(pre_diff[0])):
                self.pathOverX_count = 0

                if (abs(pre_diff[0])>abs(new_diff[0])):
                    score += 100
                elif (abs(pre_diff[0])<abs(new_diff[0])):
                    score -= 100
                else:
                    score += 0
            else:
                score += 0

                # if (self.pathOverX_count < 2):
                #     score += 50
                #     self.pathOverX_count += 1
                # else:
                #     score += 0
                #     self.pathOverX_count += 1

            if (np.sign(new_diff[1]) == np.sign(pre_diff[1])):
                self.pathOverY_count = 0

                if (abs(pre_diff[1])>abs(new_diff[1])):
                    score += 100
                elif (abs(pre_diff[1])<abs(new_diff[1])):
                    score -= 100
                else:
                    score += 0
            else:
                score += 0

                # if (self.pathOverY_count < 2):
                #     score += 50
                #     self.pathOverY_count += 1
                # else:
                #     score += 0
                #     self.pathOverY_count += 1

        else:
            score += 0

    
        return score




    

    def get_locationi_score(self,state):
        score = 0.0
        location = [state["location"]["x"],state["location"]["y"]]
        pre_location = [self.prior_game_stats["location"]["x"],self.prior_game_stats["location"]["y"]]
        map_id = state["location"]["map_id"]

        if (location != pre_location):
            if (map_id == 40):
                if (location == [5,4]):
                    score = 10
                elif (location == [5,5]):
                    score = 20
                elif (location == [5,6]):
                    score = 30
                elif (location == [5,7]):
                    score = 40
                elif (location == [5,8]):
                    score = 50
                elif (location == [5,9]):
                    score = 60
                elif (location == [5,10]):
                    score = 70
            elif (map_id == 0):
                if (location == [12,12]):
                    score = 110
                elif (location == [11,12]):
                    score = 120
                elif (location == [10,12]):
                    score = 130
                elif (location == [9,12]):
                    score = 1000
                elif (location == [9,11]):
                    score = 1100
                elif (location == [9,10]):
                    score = 1200
                elif (location == [9,9]):
                    score = 1300
                elif (location == [9,8]):
                    score = 1400
                elif (location == [9,7]):
                    score = 1500
                elif (location == [9,6]):
                    score = 2000
                elif (location == [9,5]):
                    score = 2100
                elif (location == [9,4]):
                    score = 2200
                elif (location == [9,3]):
                    score = 2300
                elif (location == [9,2]):
                    score = 2400
                elif (location == [9,1]):
                    score = 2500
        
        return score
        




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
            if (self.mapSwitch_count < 3):
                self.mapSwitch_count += 1
            else:
                self.mapSwitch_count = 0
            return_score = 10

        else:
            new_step_to_target = self.get_step_to_target(new_location,target_location)
            pre_step_to_target = self.get_step_to_target(pre_location,target_location)

            return_score = self.get_potential(new_step_to_target,pre_step_to_target,new_map)

        return return_score
    
    def get_target_location(self,map_id):
        location = [0,0]

        if (map_id == 40):
            location = [5,10]
        elif (map_id == 0):
            location = [9,0]
        else:
            location = [0,0]

        return location
    
    def get_map_change_score(self,new_map_id,pre_map_id):
        score = 0.0

        if ((new_map_id == 0) and (pre_map_id == 40)):
            score = 1000
        elif ((new_map_id == 40) and (pre_map_id == 0)):
            score = -1000
        else:
            score = 0.0

        return score
    
    def get_step_to_target(self,current_pos,target_pos):
        steps = 0

        x_step = abs(target_pos[0] - current_pos[0])
        y_step = abs(target_pos[1] - current_pos[1])
        steps = x_step + y_step

        return steps
    
    def get_potential(self,new_step,pre_step,map_id):
        new_potiential = 0
        pre_potiential = 0
        gain = 0
        score = 0.0

        if (map_id == 40):
            min_potiential = 0
            max_potiential = 10
            gain = 5
        elif (map_id == 0):
            min_potiential = 11
            max_potiential = 30
            gain = 10
        else:
            return 0
        
        max_step = max_potiential - min_potiential
        if (new_step > max_step):
            new_potiential = 0
        else:
            new_potiential = max_potiential - new_step

        if (pre_step > max_step):
            pre_potiential = 0
        else:
            pre_potiential = max_potiential - pre_step

        # if (new_potiential > pre_potiential):
        #     score = gain*new_potiential
        # elif (new_potiential < pre_potiential):
        #     score = -gain*pre_potiential
        # else:
        #     score = 0.0

        score = gain*(new_potiential - pre_potiential)

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
            return 0



    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        ifDone = 0

        if (game_stats["badges"] > self.prior_game_stats["badges"]):
            ifDone = 1
        
        if (game_stats["location"]["map_id"] == 12):
            ifDone = 1
        # Setting done to true if agent beats first gym (temporary)
        return ifDone

    def _check_if_truncated(self, game_stats: dict) -> bool:
        # Implement your truncation check logic here

        ifTruuncated = 0

        if (self.steps >= 1000):
            ifTruuncated = 1

        # if (game_stats["location"]["map_id"] == 12):
        #     ifTruuncated = 1

        # if ((game_stats["location"]["x"] == self.prior_game_stats["location"]["x"]) and (game_stats["location"]["y"] == self.prior_game_stats["location"]["y"])):
        #     self.notmove += 1

        #     if(self.notmove > 200):
        #         self.notmove = 0
        #         ifTruuncated = 1

        # Maybe if we run out of pokeballs...? or a max step count
        return ifTruuncated
