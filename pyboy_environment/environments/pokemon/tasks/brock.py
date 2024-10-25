from functools import cached_property

import numpy as np
from pyboy.utils import WindowEvent
import math

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

        self.isMapReach = [0,0,0]

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

        new_location_x = new_state["location"]["x"]
        new_location_y = new_state["location"]["y"]
        pre_location_x = self.prior_game_stats["location"]["x"]
        pre_location_y = self.prior_game_stats["location"]["y"]

        if((new_state["location"]["map_id"] == 0) and (self.prior_game_stats["location"]["map_id"] == 40) and (self.isMapReach[1] == 0)):
            return_score += 1000
            self.isMapReach[1] = 1

        if (new_state["location"]["map_id"] == 40):
            diff_y = new_location_y - pre_location_y
            return_score += diff_y*10

        if (new_state["location"]["map_id"] == 0):
            diff_dis = self.distance_to_target(new_state,[9,12])
            diff_x = new_location_x - pre_location_x

            if (diff_x != 0 ):
                return_score += 1

            return_score -= diff_dis*10

            

        return return_score
        # return new_state["badges"] - self.prior_game_stats["badges"]
    def distance_to_target(self,current_state,target):
        current_x = current_state["location"]["x"]
        current_y = current_state["location"]["y"]

        distance = math.sqrt(math.pow((target[0]-current_x),2) + math.pow((target[1]-current_y),2))
        return distance

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if agent beats first gym (temporary)
        return game_stats["badges"] > self.prior_game_stats["badges"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        # Implement your truncation check logic here

        # Maybe if we run out of pokeballs...? or a max step count
        return self.steps >= 1000
