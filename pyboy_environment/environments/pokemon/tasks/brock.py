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

        self.stepCount = 0
        self.LR_count = 0

        self.notUse = 0

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

        self.stepCount += 1
        
        used_states = [
            full_state["location"]["x"],
            full_state["location"]["y"],
            full_state["location"]["map_id"],
            self.prior_game_stats["location"]["x"],
            self.prior_game_stats["location"]["y"],
            self.prior_game_stats["location"]["map_id"],
            self.step_action
            # self.stepCount
        ]

        return used_states



    def _calculate_reward(self, new_state: dict) -> float:
        # Implement your reward calculation logic here

        return_score = 0.0
        change_10_0_gain = 2000

        if(self.mapSwitch_count1 != 0):
                if(self.mapSwitch_count1 < 5):
                    self.mapSwitch_count1 += 1
                else:
                    self.mapSwitch_count1 = 0
                    self.notUse = 1
                    # print("end_switch")

        self.notUse = 0
        if (self.notUse == 0):
            if((new_state["location"]["map_id"] == 0) and (self.prior_game_stats["location"]["map_id"] == 40)):
                return_score += change_10_0_gain
                self.mapSwitch_count1 = 1
                # print("start_switch")
            if((new_state["location"]["map_id"] == 40) and (self.prior_game_stats["location"]["map_id"] == 0)):
                return_score -= change_10_0_gain
                self.mapSwitch_count1 = 1
                # print("start_switch")

            if(self.mapSwitch_count1 == 0):
                # return_score += self.up_base_reward(new_state)
                # # print("normal")
                # return_score += self.distance_reward(new_state,10)
                # return_score += self.step_penalty(10)
                # return_score += self.collision_penalty(new_state)
                # return_score += self.inMap_step_reward(new_state)
                # return_score += self.not_move_penalty(new_state,self.prior_game_stats,2)
                # return_score += self.notOK_action_penalty(1)

                return_score += self.get_location_score(new_state,10)

                if (return_score == 0):
                    return_score += self.distance_reward(new_state,100)
                    print("in")

                return_score += self.notOK_action_penalty(10)
                
            else:
                # print("on_switch")
                return_score += 1
            
            
        else:
            print("not trained")


        if (new_state["location"]["map_id"] == 12):
            return_score += 10000
            print("YEEEEEEEEEES")
            

        # print(return_score)

        return return_score



    def up_base_reward(self,new_state):
        score = 0.0

        new_location = [new_state["location"]["x"],new_state["location"]["y"]]
        pre_location = [self.prior_game_stats["location"]["x"],self.prior_game_stats["location"]["y"]]
        new_map_id = new_state["location"]["map_id"]

        if (new_map_id != 40):
            if (new_location[1] < pre_location[1]):
                self.LR_count = 0
                score += 100
            elif (new_location[1] > pre_location[1]):
                score -= 100
            else:
                score += 0

            if (new_location[0] != pre_location[0]):
                self.LR_count += 1
                
                if (self.LR_count < 5):
                    score += 25
                else:
                    score -= 25
            else:
                score += 0
        else:
            score += 0
        
        return score
    



    def step_penalty(self,gain):
        return -gain*(self.stepCount/1000)
    
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

        if (map_id == 40):
            if ((new_location[0] == 5) and (pre_location[0] != 5)):
                score += 100
            if ((new_location[0] != 5) and (pre_location[0] == 5)):
                score -= 100

            if ((new_location[1] >= 6) and (pre_location[1] < 6)):
                score += 100
            if ((new_location[1] < 6) and (pre_location[1] >= 6)):
                score -= 100

            if ((new_location[1] >= 10) and (pre_location[1] < 10)):
                score += 100
            if ((new_location[1] < 10) and (pre_location[1] >= 10)):
                score -= 100



        if (map_id == 0):
            if ((new_location[0] < 12) and (pre_location[0] >= 12)):
                score += 500
            if ((new_location[0] >= 12) and (pre_location[0] < 12)):
                score -= 500
            if ((new_location[0] < 11) and (pre_location[0] >= 11)):
                score += 500
            if ((new_location[0] >= 11) and (pre_location[0] < 11)):
                score -= 500
            if ((new_location[0] < 10) and (pre_location[0] >= 10)):
                score += 500
            if ((new_location[0] >= 10) and (pre_location[0] < 10)):
                score -= 500
            
            if (new_location[1] > 6):
                if ((new_location[0] == 9) and (pre_location[0] != 9)):
                    score += 1000
                if ((new_location[0] != 9) and (pre_location[0] == 9)):
                    score -= 1000
            else:
                if ((new_location[0] == 10) and (pre_location[0] != 10)):
                    score += 1000
                if ((new_location[0] != 10) and (pre_location[0] == 19)):
                    score -= 1000

            if ((new_location[1] <= 9) and (pre_location[1] > 9)):
                score += 1000
            if ((new_location[1] > 9) and (pre_location[1] <= 9)):
                score -= 1000
            if ((new_location[1] <= 6) and (pre_location[1] > 6)):
                score += 1000
            if ((new_location[1] > 6) and (pre_location[1] <= 6)):
                score -= 1000
            if ((new_location[1] <= 3) and (pre_location[1] > 3)):
                score += 1000
            if ((new_location[1] > 3) and (pre_location[1] <= 3)):
                score -= 1000

        return score
    
    
    def notOK_action_penalty(self,gain):
        score = 0.0

        map_id = self.prior_game_stats["location"]["map_id"]

        if (map_id == 40):
            if ((self.step_action == 0) and (self.step_action == 1) or (self.step_action == 2) or (self.step_action == 3)):
                score += gain
            else:
                score -= gain
        elif (map_id == 0):
            if ((self.step_action == 0) and (self.step_action == 1) or (self.step_action == 2) or (self.step_action == 3)):
                score += gain
            else:
                score -= gain
        elif (map_id == 1):
            if ((self.step_action == 0) and (self.step_action == 1) or (self.step_action == 2) or (self.step_action == 3)):
                score += gain
            else:
                score -= gain

        return score

    def distance_reward(self,new_state,gain):
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
                target = [10,0]
            elif (new_map_id == 12): 
                target = [10,0]
            else:
                target = [0,0]

            new_diff = [(target[0]-new_location[0]),(target[1]-new_location[1])]
            pre_diff = [(target[0]-pre_location[0]),(target[1]-pre_location[1])]

            if (np.sign(new_diff[0]) == np.sign(pre_diff[0])):
                self.pathOverX_count = 0

                if (abs(pre_diff[0])>abs(new_diff[0])):
                    score += gain*(abs(pre_diff[0])-abs(new_diff[0]))
                elif (abs(pre_diff[0])<abs(new_diff[0])):
                    score -= gain*(abs(new_diff[0])-abs(pre_diff[0]))
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
                    score += gain*(abs(pre_diff[1])-abs(new_diff[1]))
                elif (abs(pre_diff[1])<abs(new_diff[1])):
                    score -= gain*(abs(new_diff[1])-abs(pre_diff[1]))
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
    
    def not_move_penalty(self,new_state,pre_state,gain):
        x_move = new_state["location"]["x"] - pre_state["location"]["x"]
        y_move = new_state["location"]["y"] - pre_state["location"]["y"]

        if ((x_move == 0) and (y_move == 0)):
            return -gain
            # return -gain*(self.stepCount/1000)
        else:
            return 0
            # return gain*(self.stepCount/1000)




    

    def get_location_score(self,state,gain):
        score = 0.0
        location = [state["location"]["x"],state["location"]["y"]]
        pre_location = [self.prior_game_stats["location"]["x"],self.prior_game_stats["location"]["y"]]
        map_id = state["location"]["map_id"]

        if (location != pre_location):
            if (map_id == 40):
                if ((location == [5,4]) and (pre_location != [5,4])):
                    score += 100
                if ((location != [5,4]) and (pre_location == [5,4])):
                    score -= 100

                if ((location == [5,5]) and (pre_location != [5,5])):
                    score += 200
                if ((location != [5,5]) and (pre_location == [5,5])):
                    score -= 200

                if ((location == [5,6]) and (pre_location != [5,6])):
                    score += 300
                if ((location != [5,6]) and (pre_location == [5,6])):
                    score -= 300
                
                if ((location == [5,7]) and (pre_location != [5,7])):
                    score += 400
                if ((location != [5,7]) and (pre_location == [5,7])):
                    score -= 400

                if ((location == [5,8]) and (pre_location != [5,8])):
                    score += 500
                if ((location != [5,8]) and (pre_location == [5,8])):
                    score -= 500
                
                if ((location == [5,9]) and (pre_location != [5,9])):
                    score += 600
                if ((location != [5,9]) and (pre_location == [5,9])):
                    score -= 600
                
                if ((location == [5,10]) and (pre_location != [5,10])):
                    score += 700
                if ((location != [5,10]) and (pre_location == [5,10])):
                    score -= 700

                if ((location == [5,11]) and (pre_location != [5,11])):
                    score += 800
                if ((location != [5,11]) and (pre_location == [5,11])):
                    score -= 800

            elif (map_id == 0):
                if ((location == [11,12]) and (pre_location != [11,12])):
                    score += 1100
                if ((location != [11,12]) and (pre_location == [11,12])):
                    score -= 1100

                if ((location == [10,12]) and (pre_location != [10,121])):
                    score += 1200
                if ((location != [10,12]) and (pre_location == [10,12])):
                    score -= 1200

                if ((location == [9,12]) and (pre_location != [9,12])):
                    score += 1300
                if ((location != [9,12]) and (pre_location == [9,12])):
                    score -= 1300
                
                if ((location == [8,12]) and (pre_location != [8,12])):
                    score += 1400
                if ((location != [8,12]) and (pre_location == [8,12])):
                    score -= 1400

                if ((location == [8,11]) and (pre_location != [8,11])):
                    score += 1500
                if ((location != [8,11]) and (pre_location == [8,11])):
                    score -= 1500

                if ((location == [8,10]) and (pre_location != [8,10])):
                    score += 1600
                if ((location != [8,10]) and (pre_location == [8,10])):
                    score -= 1600

                if ((location == [8,9]) and (pre_location != [8,9])):
                    score += 1700
                if ((location != [8,9]) and (pre_location == [8,9])):
                    score -= 1700

                if ((location == [8,8]) and (pre_location != [8,8])):
                    score += 1800
                if ((location != [8,8]) and (pre_location == [8,8])):
                    score -= 1800

                if ((location == [8,7]) and (pre_location != [8,7])):
                    score += 1900
                if ((location != [8,7]) and (pre_location == [8,7])):
                    score -= 1900

                if ((location == [8,6]) and (pre_location != [8,6])):
                    score += 2000
                if ((location != [8,6]) and (pre_location == [8,6])):
                    score -= 2000

                if ((location == [9,6]) and (pre_location != [9,6])):
                    score += 2100
                if ((location != [9,6]) and (pre_location == [9,6])):
                    score -= 2100

                if ((location == [10,6]) and (pre_location != [10,6])):
                    score += 2200
                if ((location != [10,6]) and (pre_location == [10,6])):
                    score -= 2200

                if ((location == [10,5]) and (pre_location != [10,5])):
                    score += 2300
                if ((location != [10,5]) and (pre_location == [10,5])):
                    score -= 2300

                if ((location == [10,4]) and (pre_location != [10,4])):
                    score += 2400
                if ((location != [10,4]) and (pre_location == [10,4])):
                    score -= 2400

                if ((location == [10,3]) and (pre_location != [10,3])):
                    score += 2500
                if ((location != [10,3]) and (pre_location == [10,3])):
                    score -= 2500

                if ((location == [10,2]) and (pre_location != [10,2])):
                    score += 2600
                if ((location != [10,2]) and (pre_location == [10,2])):
                    score -= 2600
                
                if ((location == [10,1]) and (pre_location != [10,1])):
                    score += 2700
                if ((location != [10,1]) and (pre_location == [10,1])):
                    score -= 2700
                
                if ((location == [10,0]) and (pre_location != [10,0])):
                    score += 2800
                if ((location != [10,0]) and (pre_location == [10,0])):
                    score -= 2800
        
        return gain*score
        




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
            location = [5,12]
        elif (map_id == 0):
            location = [8,0]
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



    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        ifDone = 0

        if (game_stats["badges"] > self.prior_game_stats["badges"]):
            ifDone = 1
        
        if (game_stats["location"]["map_id"] == 1):
            ifDone = 1

        if (ifDone):
            self.stepCount = 0
            self.notUse = 1
        # Setting done to true if agent beats first gym (temporary)
        return ifDone

    def _check_if_truncated(self, game_stats: dict) -> bool:
        # Implement your truncation check logic here

        ifTruuncated = 0

        if (self.steps >= 1000):
            ifTruuncated = 1

        # if (game_stats["location"]["map_id"] == 12):
        #     ifTruuncated = 1

        # if (game_stats["location"]["map_id"] != 40):
        #     if ((game_stats["location"]["x"] == self.prior_game_stats["location"]["x"]) and (game_stats["location"]["y"] == self.prior_game_stats["location"]["y"])):
        #         self.notmove += 1

        #         if(self.notmove > 200):
        #             self.notmove = 0
        #             ifTruuncated = 1
        #     else:
        #         self.notmove = 0

        # Maybe if we run out of pokeballs...? or a max step count
        if (ifTruuncated):
            self.notmove = 0
            self.stepCount = 0
            self.notUse = 1

        return ifTruuncated
