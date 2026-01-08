import gymnasium as gym
import numpy as np
from typing import Optional, Callable, Dict, Any
from core.game import BuckshotRouletteGame
from core.constants import (
    Item,
    Turn,
    ACTION_MAP,
    GAME_ACTIONS,
    OBS_SIZE,
    MAX_HP,
    MAX_ITEM_COUNT,
    MAX_CYLINDER,
)


class BuckshotRouletteEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        opponent_policy: Optional[Callable] = None,
        force_agent_as_player: Optional[bool] = None,
    ):
        super().__init__()

        self.opponent_policy = opponent_policy
        self.force_agent_as_player = force_agent_as_player
        self.game = BuckshotRouletteGame()

        self.action_space = gym.spaces.Discrete(len(GAME_ACTIONS))
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )

        self._agent_is_player = True
        self._agent_went_first = False
        self._episode_steps = 0
        self._max_episode_steps = 1000

        # Optimization: Pre-allocation & Caching
        self._obs_array = np.zeros(OBS_SIZE, dtype=np.float32)
        self._action_mask = np.zeros(len(GAME_ACTIONS), dtype=np.int8)
        self._max_hp_inv = 1.0 / MAX_HP
        self._max_item_inv = 1.0 / MAX_ITEM_COUNT
        self._max_cylinder_inv = 1.0 / MAX_CYLINDER
        self._handcuff_inv = 1.0 / 2.0

    def _get_obs(self, for_opponent: bool = False) -> np.ndarray:
        if for_opponent:
            bot = self.game.dealer if self._agent_is_player else self.game.player
            target = self.game.player if self._agent_is_player else self.game.dealer
        else:
            bot = self.game.player if self._agent_is_player else self.game.dealer
            target = self.game.dealer if self._agent_is_player else self.game.player

        bot_items = bot.items
        target_items = target.items
        obs = self._obs_array

        # Bot stats
        obs[0] = bot.hp * self._max_hp_inv
        obs[1] = bot_items.count(Item.GLASS) * self._max_item_inv
        obs[2] = bot_items.count(Item.CIGARETTES) * self._max_item_inv
        obs[3] = bot_items.count(Item.HANDCUFFS) * self._max_item_inv
        obs[4] = bot_items.count(Item.SAW) * self._max_item_inv
        obs[5] = bot_items.count(Item.BEER) * self._max_item_inv

        # Target stats
        obs[6] = target.hp * self._max_hp_inv
        obs[7] = target_items.count(Item.GLASS) * self._max_item_inv
        obs[8] = target_items.count(Item.CIGARETTES) * self._max_item_inv
        obs[9] = target_items.count(Item.HANDCUFFS) * self._max_item_inv
        obs[10] = target_items.count(Item.SAW) * self._max_item_inv
        obs[11] = target_items.count(Item.BEER) * self._max_item_inv
        obs[12] = target.handcuff_strength * self._handcuff_inv

        # Bullet info
        bullet_seq = self.game.bullet_sequence
        blanks_left = bullet_seq.count(0)
        lives_left = len(bullet_seq) - blanks_left

        obs[13] = blanks_left * self._max_cylinder_inv
        obs[14] = lives_left * self._max_cylinder_inv

        # Next bullet knowledge
        if bot.known_next and len(bullet_seq) > 0:
            obs[15] = 1.0 if bullet_seq[0] == 1 else 0.0  # Live
            obs[16] = 1.0 if bullet_seq[0] == 0 else 0.0  # Blank
            obs[17] = 0.0
        else:
            obs[15] = 0.0
            obs[16] = 0.0
            obs[17] = 1.0  # Unknown

        obs[18] = 1.0 if self.game.saw_active else 0.0

        return obs

    def action_masks(self) -> np.ndarray:
        return self.game.get_valid_actions_mask()

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> tuple:
        super().reset(seed=seed)

        if seed is None:
            seed = np.random.randint(0, 2**31 - 1)

        np.random.seed(seed)
        rng = np.random.default_rng(seed)

        self.game = BuckshotRouletteGame(rng_seed=seed)
        self.game.start_new_round()

        # Role assignment
        if self.force_agent_as_player is not None:
            self._agent_is_player = self.force_agent_as_player
        else:
            self._agent_is_player = rng.choice([True, False])

        # Turn order
        agent_goes_first = rng.choice([True, False])
        self._agent_went_first = agent_goes_first

        if self._agent_is_player:
            if agent_goes_first:
                self.game.turn = Turn.PLAYER
            else:
                self.game.turn = Turn.DEALER
                self._opponent_turn()
        else:
            if agent_goes_first:
                self.game.turn = Turn.DEALER
            else:
                self.game.turn = Turn.PLAYER
                self._opponent_turn()

        self._episode_steps = 0
        return self._get_obs(), {}

    def _opponent_turn(self):
        max_opponent_steps = 500
        opponent_steps = 0

        while (
            not self._is_agent_turn()
            and not self._is_terminal()
            and opponent_steps < max_opponent_steps
        ):
            action_mask = self.action_masks()

            if self.opponent_policy is not None:
                obs = self._get_obs(for_opponent=True)
                action_idx = self.opponent_policy(obs, action_mask)
            else:
                valid_actions = np.where(action_mask == 1)[0]
                if len(valid_actions) == 0:
                    break
                action_idx = np.random.choice(valid_actions)

            action = ACTION_MAP[action_idx]
            self.game.step(action)
            opponent_steps += 1

    def _is_agent_turn(self) -> bool:
        return (
            self.game.turn == Turn.PLAYER
            if self._agent_is_player
            else self.game.turn == Turn.DEALER
        )

    def _is_terminal(self) -> bool:
        return self.game.player.hp <= 0 or self.game.dealer.hp <= 0

    def step(self, action_idx: int) -> tuple:
        self._episode_steps += 1
        action = ACTION_MAP[action_idx]
        step_result = self.game.step(action)
        terminated = step_result.terminated
        agent_hp_change = step_result.new_bot_hp - step_result.prev_bot_hp
        opponent_hp_change = step_result.new_target_hp - step_result.prev_target_hp

        # Calculate Reward
        if not step_result.valid:
            reward = -10.0
        else:
            reward = 0.0
            if opponent_hp_change < 0:
                reward += -opponent_hp_change
            if agent_hp_change < 0:
                reward += agent_hp_change

        if terminated:
            # Win = +100, Loss = -100, Draw/Neither = 0
            outcome = 0
            if step_result.dealer_dead and not step_result.player_dead:
                outcome = 100.0
            elif step_result.player_dead:
                outcome = -100.0

            multiplier = 1.0 if self._agent_is_player else -1.0
            reward += outcome * multiplier
        else:
            self._opponent_turn()
            # Check if opponent ended the game
            if self._is_terminal():
                terminated = True
                player_hp = self.game.player.hp
                dealer_hp = self.game.dealer.hp

                if self._agent_is_player:
                    reward += 100.0 if dealer_hp <= 0 else -100.0
                else:
                    reward += 100.0 if player_hp <= 0 else -100.0

        truncated = self._episode_steps >= self._max_episode_steps
        info = {
            "invalid_action": not step_result.valid,
            "episode_steps": self._episode_steps,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        print(f"\n=== Round {self.game.round}, Subround {self.game.sub_round} ===")
        print(f"Player HP: {self.game.player.hp} | Dealer HP: {self.game.dealer.hp}")
        print(f"Turn: {self.game.turn.name}")
        print(
            f"Bullets: {len(self.game.bullet_sequence)} "
            f"({self.game.bullet_sequence.count(1)} live, {self.game.bullet_sequence.count(0)} blank)"
        )
        print(f"Player items: {[item.name for item in self.game.player.items]}")
        print(f"Dealer items: {[item.name for item in self.game.dealer.items]}")
        print(f"Saw active: {self.game.saw_active}")
