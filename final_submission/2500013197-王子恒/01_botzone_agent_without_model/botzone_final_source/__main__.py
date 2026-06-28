import os
import sys

import numpy as np
import torch

from feature_v2 import FeatureAgent
from model_v6 import CNNModel


DEFAULT_MODEL_PATH = os.path.join("data", "mahjong_v7_calibrated_best.pkl")
CALL_BIAS = 1.50
CHI_EXTRA_BIAS = 0.35
PENG_EXTRA_BIAS = 0.35


def compatible_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_policy(path):
    torch.set_num_threads(1)
    model = CNNModel()
    state = compatible_torch_load(path, map_location=torch.device("cpu"))
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.eval()
    return model


def apply_reaction_bias(logits, action_mask):
    pass_offset = FeatureAgent.OFFSET_ACT["Pass"]
    chi_offset = FeatureAgent.OFFSET_ACT["Chi"]
    peng_offset = FeatureAgent.OFFSET_ACT["Peng"]
    gang_offset = FeatureAgent.OFFSET_ACT["Gang"]
    angang_offset = FeatureAgent.OFFSET_ACT["AnGang"]

    mask = action_mask[0]
    can_respond = bool(mask[pass_offset].item() > 0) and (
        bool(mask[chi_offset:peng_offset].max().item() > 0)
        or bool(mask[peng_offset:gang_offset].max().item() > 0)
        or bool(mask[gang_offset:angang_offset].max().item() > 0)
    )
    if not can_respond:
        return logits

    adjusted = logits.clone()
    adjusted[:, chi_offset:peng_offset] += CALL_BIAS + CHI_EXTRA_BIAS
    adjusted[:, peng_offset:gang_offset] += CALL_BIAS + PENG_EXTRA_BIAS
    adjusted[:, gang_offset:angang_offset] += CALL_BIAS
    return adjusted


def obs2response(model, agent, observation):
    observation_tensor = torch.from_numpy(
        np.expand_dims(observation["observation"], axis=0)
    )
    mask_tensor = torch.from_numpy(
        np.expand_dims(observation["action_mask"], axis=0)
    )
    with torch.no_grad():
        logits = model(
            {
                "is_training": False,
                "obs": {
                    "observation": observation_tensor,
                    "action_mask": mask_tensor,
                },
            }
        )
        logits = apply_reaction_bias(logits, mask_tensor)
        action = int(logits.argmax(dim=1).item())
    if observation["action_mask"][action] != 1:
        raise RuntimeError("model selected an illegal action")
    return agent.action2response(action)


class BotzoneRunner:
    def __init__(self, model):
        self.model = model
        self.agent = None
        self.seat_wind = None
        self.pending_concealed_kong = None
        self.last_event_was_draw = False

    def _require_agent(self):
        if self.agent is None:
            raise RuntimeError("Botzone seat request has not been received")

    def _choose(self, observation):
        return obs2response(self.model, self.agent, observation)

    def handle_request(self, request):
        tokens = request.strip().split()
        if not tokens:
            raise ValueError("empty Botzone request")

        request_type = tokens[0]
        if request_type == "0":
            return self._handle_start(tokens)
        self._require_agent()
        if request_type == "1":
            return self._handle_deal(tokens)
        if request_type == "2":
            return self._handle_self_draw(tokens)
        if request_type == "3":
            return self._handle_player_event(tokens)
        raise ValueError("unknown Botzone request: %s" % request)

    def _handle_start(self, tokens):
        if len(tokens) < 3:
            raise ValueError("invalid start request")
        self.seat_wind = int(tokens[1])
        self.agent = FeatureAgent(self.seat_wind)
        self.agent.request2obs("Wind %s" % tokens[2])
        self.pending_concealed_kong = None
        self.last_event_was_draw = False
        return "PASS"

    def _handle_deal(self, tokens):
        if len(tokens) < 18:
            raise ValueError("invalid deal request")
        self.agent.request2obs("Deal " + " ".join(tokens[5:]))
        return "PASS"

    def _handle_self_draw(self, tokens):
        if len(tokens) != 2:
            raise ValueError("invalid self draw request")
        observation = self.agent.request2obs("Draw %s" % tokens[1])
        response = self._choose(observation)
        parts = response.split()
        self.last_event_was_draw = True

        if parts[0] == "Hu":
            return "HU"
        if parts[0] == "Play":
            return "PLAY %s" % parts[1]
        if parts[0] == "Gang":
            if len(parts) != 2:
                raise RuntimeError("self draw Gang must include a tile")
            self.pending_concealed_kong = parts[1]
            return "GANG %s" % parts[1]
        if parts[0] == "BuGang":
            return "BUGANG %s" % parts[1]
        raise RuntimeError("invalid action after self draw: %s" % response)

    def _handle_player_event(self, tokens):
        if len(tokens) < 3:
            raise ValueError("invalid player event")
        player = int(tokens[1])
        event = tokens[2].upper()

        if event == "DRAW":
            self.agent.request2obs("Player %d Draw" % player)
            self.last_event_was_draw = True
            return "PASS"

        if event == "GANG":
            if player == self.seat_wind and self.pending_concealed_kong:
                self.agent.request2obs(
                    "Player %d AnGang %s" % (player, self.pending_concealed_kong)
                )
                self.pending_concealed_kong = None
            elif self.last_event_was_draw:
                self.agent.request2obs("Player %d AnGang" % player)
            else:
                self.agent.request2obs("Player %d Gang" % player)
            self.last_event_was_draw = False
            return "PASS"

        if event == "BUGANG":
            if len(tokens) < 4:
                raise ValueError("BUGANG event requires a tile")
            observation = self.agent.request2obs(
                "Player %d BuGang %s" % (player, tokens[3])
            )
            self.last_event_was_draw = False
            if player == self.seat_wind:
                return "PASS"
            response = self._choose(observation)
            return "HU" if response == "Hu" else "PASS"

        if event == "HU":
            self.agent.request2obs("Player %d Hu" % player)
            self.last_event_was_draw = False
            return "PASS"

        if event == "INVALID":
            self.agent.request2obs("Player %d Invalid" % player)
            self.last_event_was_draw = False
            return "PASS"

        if event not in ("PLAY", "CHI", "PENG"):
            raise ValueError("unsupported player event: %s" % event)

        self.last_event_was_draw = False
        if event == "CHI":
            if len(tokens) < 5:
                raise ValueError("CHI event requires center and discard")
            self.agent.request2obs("Player %d Chi %s" % (player, tokens[3]))
        elif event == "PENG":
            self.agent.request2obs("Player %d Peng" % player)

        discarded_tile = tokens[-1]
        observation = self.agent.request2obs(
            "Player %d Play %s" % (player, discarded_tile)
        )
        if player == self.seat_wind:
            return "PASS"

        response = self._choose(observation)
        parts = response.split()
        if parts[0] == "Hu":
            return "HU"
        if parts[0] == "Pass":
            return "PASS"
        if parts[0] == "Gang":
            self.pending_concealed_kong = None
            return "GANG"
        if parts[0] in ("Peng", "Chi"):
            return self._prepare_meld_response(response)
        raise RuntimeError("invalid reaction action: %s" % response)

    def _prepare_meld_response(self, response):
        observation = self.agent.request2obs(
            "Player %d %s" % (self.seat_wind, response)
        )
        discard_response = self._choose(observation)
        if not discard_response.startswith("Play "):
            discard_response = self._first_legal_discard(observation)
        discard_tile = discard_response.split()[1]

        if response.startswith("Chi "):
            center_tile = response.split()[1]
            output = "CHI %s %s" % (center_tile, discard_tile)
            self.agent.request2obs(
                "Player %d UnChi %s" % (self.seat_wind, center_tile)
            )
        else:
            output = "PENG %s" % discard_tile
            self.agent.request2obs("Player %d UnPeng" % self.seat_wind)
        return output

    def _first_legal_discard(self, observation):
        play_offset = FeatureAgent.OFFSET_ACT["Play"]
        chi_offset = FeatureAgent.OFFSET_ACT["Chi"]
        for action in range(play_offset, chi_offset):
            if observation["action_mask"][action] == 1:
                return self.agent.action2response(action)
        raise RuntimeError("meld succeeded but no legal discard exists")


def _emit_response(response):
    print(response)
    print(">>>BOTZONE_REQUEST_KEEP_RUNNING<<<")
    sys.stdout.flush()


def run_botzone(model_path):
    model = load_policy(model_path)
    runner = BotzoneRunner(model)

    if sys.stdin.readline() == "":
        return

    while True:
        request = sys.stdin.readline()
        if request == "":
            return
        if not request.strip():
            continue
        response = runner.handle_request(request)
        _emit_response(response)


if __name__ == "__main__":
    model_path = os.environ.get("MAHJONG_MODEL_PATH", DEFAULT_MODEL_PATH)
    run_botzone(model_path)
