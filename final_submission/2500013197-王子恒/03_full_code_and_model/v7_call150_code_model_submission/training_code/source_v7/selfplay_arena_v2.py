import argparse
import importlib
import json
import random
from collections import Counter, defaultdict

import numpy as np
import torch

from feature_v2 import FeatureAgent
from MahjongGB import MahjongShanten


TILES = FeatureAgent.TILE_LIST[:34]
PASS = FeatureAgent.OFFSET_ACT["Pass"]
HU = FeatureAgent.OFFSET_ACT["Hu"]
PLAY_BEGIN = FeatureAgent.OFFSET_ACT["Play"]
PLAY_END = FeatureAgent.OFFSET_ACT["Chi"]
CHI_BEGIN = FeatureAgent.OFFSET_ACT["Chi"]
PENG_BEGIN = FeatureAgent.OFFSET_ACT["Peng"]
GANG_BEGIN = FeatureAgent.OFFSET_ACT["Gang"]
ANGANG_BEGIN = FeatureAgent.OFFSET_ACT["AnGang"]
BUGANG_BEGIN = FeatureAgent.OFFSET_ACT["BuGang"]


def action_kind(action):
    if action == PASS:
        return "Pass"
    if action == HU:
        return "Hu"
    if PLAY_BEGIN <= action < CHI_BEGIN:
        return "Play"
    if CHI_BEGIN <= action < PENG_BEGIN:
        return "Chi"
    if PENG_BEGIN <= action < GANG_BEGIN:
        return "Peng"
    if GANG_BEGIN <= action < ANGANG_BEGIN:
        return "Gang"
    if ANGANG_BEGIN <= action < BUGANG_BEGIN:
        return "AnGang"
    return "BuGang"


class Policy:
    def __init__(self, name, model, device, call_margin=0.0, allow_chi=True, allow_peng=True, allow_gang=True):
        self.name = name
        self.model = model
        self.device = device
        self.call_margin = call_margin
        self.allow_chi = allow_chi
        self.allow_peng = allow_peng
        self.allow_gang = allow_gang
        self.action_counts = Counter()

    def logits(self, obs):
        with torch.no_grad():
            input_dict = {
                "is_training": False,
                "obs": {
                    "observation": torch.as_tensor(
                        np.expand_dims(obs["observation"], 0),
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    "action_mask": torch.as_tensor(
                        np.expand_dims(obs["action_mask"], 0),
                        dtype=torch.float32,
                        device=self.device,
                    ),
                },
            }
            return self.model(input_dict).detach().cpu().numpy().reshape(-1)

    def select_action(self, agent, obs, mode):
        mask = obs["action_mask"]
        if mask[HU] > 0:
            self.action_counts["Hu"] += 1
            return HU, "Hu"

        logits = self.logits(obs)
        action = int(logits.argmax())
        kind = action_kind(action)

        if mode == "respond":
            if kind == "Chi" and not self.allow_chi:
                action, kind = PASS, "Pass"
            elif kind == "Peng" and not self.allow_peng:
                action, kind = PASS, "Pass"
            elif kind == "Gang" and not self.allow_gang:
                action, kind = PASS, "Pass"
            elif kind in ("Chi", "Peng", "Gang") and self.call_margin > 0:
                if logits[action] - logits[PASS] < self.call_margin:
                    action, kind = PASS, "Pass"

        if mask[action] <= 0:
            valid = np.nonzero(mask)[0]
            play = [int(a) for a in valid if PLAY_BEGIN <= a < PLAY_END]
            action = play[0] if mode == "play" and play else int(valid[0])
            kind = action_kind(action)

        self.action_counts[kind] += 1
        return action, agent.action2response(action)


class Arena:
    def __init__(self, policies, seed, prevalent_wind=0, max_turns=400):
        self.policies = policies
        self.rng = random.Random(seed)
        self.prevalent_wind = prevalent_wind
        self.max_turns = max_turns
        self.agents = [FeatureAgent(i) for i in range(4)]
        self.wall = []
        self.current = 0
        self.turns = 0
        self.events = []

    def emit(self, request):
        for agent in self.agents:
            agent.request2obs(request)
        self.events.append(request)

    def init_game(self):
        wall = [tile for tile in TILES for _ in range(4)]
        self.rng.shuffle(wall)
        hands = [wall[i * 13:(i + 1) * 13] for i in range(4)]
        self.wall = wall[52:]
        for agent in self.agents:
            agent.request2obs("Wind %d" % self.prevalent_wind)
        for i, hand in enumerate(hands):
            self.agents[i].request2obs("Deal " + " ".join(hand))
        self.current = 0

    def apply_draw_to_agents(self, player, tile):
        obs = None
        for i, agent in enumerate(self.agents):
            if i == player:
                obs = agent.request2obs("Draw %s" % tile)
            else:
                agent.request2obs("Player %d Draw" % player)
        self.events.append("Draw %d %s" % (player, tile))
        return obs

    def broadcast_play(self, player, tile):
        responses = {}
        for i, agent in enumerate(self.agents):
            obs = agent.request2obs("Player %d Play %s" % (player, tile))
            if i != player:
                responses[i] = obs
        self.events.append("Player %d Play %s" % (player, tile))
        return responses

    def broadcast_meld(self, player, meld_response):
        parts = meld_response.split()
        meld = parts[0]
        request = "Player %d %s" % (player, " ".join(parts))
        obs_after = None
        for i, agent in enumerate(self.agents):
            obs = agent.request2obs(request)
            if i == player:
                obs_after = obs
        self.events.append(request)
        return meld, obs_after

    def calc_shanten(self, seat):
        if seat is None:
            return None
        try:
            return int(MahjongShanten(tuple(self.agents[seat].packs[0]), tuple(self.agents[seat].hand)))
        except Exception:
            return 99

    def finish(self, winner=None, loser=None, reason="draw"):
        shanten = [self.calc_shanten(i) for i in range(4)]
        if winner is not None:
            shanten[winner] = -1
        result = {
            "winner": winner,
            "loser": loser,
            "reason": reason,
            "turns": self.turns,
            "policy_by_seat": [p.name for p in self.policies],
            "shanten_by_seat": shanten,
        }
        return result

    def resolve_discard(self, player, tile):
        responses = self.broadcast_play(player, tile)

        hu_players = []
        calls = []
        for offset in range(1, 4):
            p = (player + offset) % 4
            obs = responses[p]
            action, response = self.policies[p].select_action(self.agents[p], obs, "respond")
            kind = action_kind(action)
            if kind == "Hu":
                hu_players.append(p)
            elif kind in ("Gang", "Peng", "Chi"):
                calls.append((kind, p, action, response))

        if hu_players:
            return self.finish(winner=hu_players[0], loser=player, reason="ron")

        gang_peng = [c for c in calls if c[0] in ("Gang", "Peng")]
        if gang_peng:
            priority = {"Gang": 0, "Peng": 1}
            gang_peng.sort(key=lambda c: (priority[c[0]], (c[1] - player) % 4))
            kind, p, action, response = gang_peng[0]
            meld, obs_after = self.broadcast_meld(p, response)
            if kind == "Gang":
                self.current = p
                return None
            return self.play_from_obs(p, obs_after)

        next_player = (player + 1) % 4
        chi_calls = [c for c in calls if c[0] == "Chi" and c[1] == next_player]
        if chi_calls:
            kind, p, action, response = chi_calls[0]
            meld, obs_after = self.broadcast_meld(p, response)
            return self.play_from_obs(p, obs_after)

        self.current = (player + 1) % 4
        return None

    def play_from_obs(self, player, obs):
        action, response = self.policies[player].select_action(self.agents[player], obs, "play")
        parts = response.split()
        if parts[0] == "Hu":
            return self.finish(winner=player, loser=None, reason="tsumo")
        if parts[0] == "Play":
            return self.resolve_discard(player, parts[1])
        if parts[0] == "Gang" and len(parts) > 1:
            request = "Player %d AnGang %s" % (player, parts[1])
            for agent in self.agents:
                agent.request2obs(request if agent.seatWind == player else "Player %d AnGang" % player)
            self.events.append(request)
            self.current = player
            return None
        if parts[0] == "BuGang":
            request = "Player %d BuGang %s" % (player, parts[1])
            rob_obs = {}
            for i, agent in enumerate(self.agents):
                obs2 = agent.request2obs(request)
                if i != player:
                    rob_obs[i] = obs2
            for offset in range(1, 4):
                p = (player + offset) % 4
                action2, response2 = self.policies[p].select_action(self.agents[p], rob_obs[p], "respond")
                if action_kind(action2) == "Hu":
                    return self.finish(winner=p, loser=player, reason="rob_kong")
            self.current = player
            return None
        valid = np.nonzero(obs["action_mask"])[0]
        plays = [int(a) for a in valid if PLAY_BEGIN <= a < PLAY_END]
        if plays:
            return self.resolve_discard(player, self.agents[player].TILE_LIST[plays[0] - PLAY_BEGIN])
        self.current = (player + 1) % 4
        return None

    def run(self):
        self.init_game()
        while self.wall and self.turns < self.max_turns:
            player = self.current
            tile = self.wall.pop()
            self.turns += 1
            obs = self.apply_draw_to_agents(player, tile)
            result = self.play_from_obs(player, obs)
            if result is not None:
                return result
        return self.finish(reason="huang")


def make_policy(name, model, device):
    if name == "base":
        return Policy(name, model, device)
    if name == "no_chi":
        return Policy(name, model, device, allow_chi=False)
    if name == "no_calls":
        return Policy(name, model, device, allow_chi=False, allow_peng=False, allow_gang=False)
    if name.startswith("margin"):
        margin = float(name.replace("margin", ""))
        return Policy(name, model, device, call_margin=margin)
    raise ValueError("unknown policy %s" % name)


def run_match(base_model, candidate_model, device, candidate, seed, seat):
    policies = [make_policy("base", base_model, device) for _ in range(4)]
    policies[seat] = make_policy(candidate, candidate_model, device)
    arena = Arena(policies, seed)
    return arena.run(), policies


def load_model(module_name, ckpt, device):
    module = importlib.import_module(module_name)
    model = module.CNNModel().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="final_v2_best.pkl")
    parser.add_argument("--base-ckpt", default="")
    parser.add_argument("--candidate-ckpt", default="")
    parser.add_argument("--base-model-module", default="model_v3")
    parser.add_argument("--candidate-model-module", default="")
    parser.add_argument("--candidates", default="base,no_chi,margin0.5,margin1.0,margin2.0,no_calls")
    parser.add_argument("--games", type=int, default=80)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    base_ckpt = args.base_ckpt or args.ckpt
    candidate_ckpt = args.candidate_ckpt or args.ckpt
    candidate_module = args.candidate_model_module or args.base_model_module
    base_model = load_model(args.base_model_module, base_ckpt, device)
    if candidate_ckpt == base_ckpt and candidate_module == args.base_model_module:
        candidate_model = base_model
    else:
        candidate_model = load_model(candidate_module, candidate_ckpt, device)

    candidates = [x.strip() for x in args.candidates.split(",") if x.strip()]
    all_results = {}
    for cand in candidates:
        stats = Counter()
        action_counts = Counter()
        for g in range(args.games):
            seat = g % 4
            seed = args.seed + g
            try:
                result, policies = run_match(base_model, candidate_model, device, cand, seed, seat)
            except Exception as exc:
                stats["errors"] += 1
                if stats["errors"] <= 3:
                    print("ERROR", cand, g, repr(exc))
                continue
            winner = result["winner"]
            if winner is None:
                stats["draw"] += 1
            elif winner == seat:
                stats["candidate_win"] += 1
            else:
                stats["base_win"] += 1
            stats[result["reason"]] += 1
            stats["turns"] += result["turns"]
            shanten = result["shanten_by_seat"]
            stats["candidate_shanten"] += shanten[seat]
            base_shanten = [shanten[i] for i in range(4) if i != seat]
            stats["base_shanten"] += sum(base_shanten) / 3.0
            if shanten[seat] < min(base_shanten):
                stats["best_shanten"] += 1
            if shanten[seat] <= min(base_shanten):
                stats["tied_best_shanten"] += 1
            action_counts.update(policies[seat].action_counts)
        played = max(args.games - stats["errors"], 1)
        all_results[cand] = {
            "games": played,
            "candidate_win": stats["candidate_win"],
            "base_win": stats["base_win"],
            "draw": stats["draw"],
            "win_rate": stats["candidate_win"] / played,
            "non_draw_win_share": stats["candidate_win"] / max(stats["candidate_win"] + stats["base_win"], 1),
            "avg_turns": stats["turns"] / played,
            "candidate_avg_shanten": stats["candidate_shanten"] / played,
            "base_avg_shanten": stats["base_shanten"] / played,
            "shanten_edge": stats["base_shanten"] / played - stats["candidate_shanten"] / played,
            "best_shanten_rate": stats["best_shanten"] / played,
            "tied_best_shanten_rate": stats["tied_best_shanten"] / played,
            "reasons": {k: stats[k] for k in ("tsumo", "ron", "rob_kong", "huang")},
            "errors": stats["errors"],
            "candidate_action_counts": dict(action_counts),
        }
        print("CANDIDATE", cand, json.dumps(all_results[cand], sort_keys=True))

    print(json.dumps(all_results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
