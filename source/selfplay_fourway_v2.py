import argparse
import itertools
import json
import sys
from collections import Counter, defaultdict

import torch

from selfplay_arena_v2 import Arena, Policy, load_model


def parse_player(spec):
    try:
        name, rest = spec.split("=", 1)
        module, ckpt = rest.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "player must look like name=model_module:checkpoint"
        ) from exc
    name = name.strip()
    module = module.strip()
    ckpt = ckpt.strip()
    if not name or not module or not ckpt:
        raise argparse.ArgumentTypeError(
            "player must look like name=model_module:checkpoint"
        )
    return {"name": name, "module": module, "ckpt": ckpt}


def empty_player_stats():
    return {
        "games": 0,
        "wins": 0,
        "draws": 0,
        "deal_in": 0,
        "tsumo_wins": 0,
        "ron_wins": 0,
        "rob_kong_wins": 0,
        "turns_sum": 0,
        "final_shanten_sum": 0.0,
        "non_win_final_shanten_sum": 0.0,
        "non_win_final_shanten_count": 0,
        "best_shanten": 0,
        "tied_best_shanten": 0,
        "action_counts": Counter(),
        "seat_games": Counter(),
        "seat_wins": Counter(),
    }


def serializable_stats(stats, total_games, total_non_draw):
    out = {}
    for name, s in stats.items():
        games = max(s["games"], 1)
        wins = s["wins"]
        non_win_count = max(s["non_win_final_shanten_count"], 1)
        actions = dict(sorted(s["action_counts"].items()))
        calls = sum(actions.get(k, 0) for k in ("Chi", "Peng", "Gang"))
        out[name] = {
            "games": s["games"],
            "wins": wins,
            "draws": s["draws"],
            "win_rate": wins / games,
            "non_draw_win_share": wins / max(total_non_draw, 1),
            "expected_non_draw_share": 0.25,
            "draw_rate": s["draws"] / games,
            "deal_in": s["deal_in"],
            "deal_in_rate": s["deal_in"] / games,
            "tsumo_wins": s["tsumo_wins"],
            "ron_wins": s["ron_wins"],
            "rob_kong_wins": s["rob_kong_wins"],
            "avg_turns": s["turns_sum"] / games,
            "avg_final_shanten": s["final_shanten_sum"] / games,
            "avg_non_win_final_shanten": s["non_win_final_shanten_sum"] / non_win_count,
            "best_shanten_rate": s["best_shanten"] / games,
            "tied_best_shanten_rate": s["tied_best_shanten"] / games,
            "action_counts": actions,
            "calls": calls,
            "calls_per_game": calls / games,
            "seat_games": {str(k): v for k, v in sorted(s["seat_games"].items())},
            "seat_wins": {str(k): v for k, v in sorted(s["seat_wins"].items())},
            "seat_win_rates": {
                str(k): s["seat_wins"][k] / max(s["seat_games"][k], 1)
                for k in sorted(s["seat_games"])
            },
        }
    return out


def run_fourway(players, games, seed, device, save_games=False, progress_every=0):
    loaded = {}
    for player in players:
        loaded[player["name"]] = {
            "model": load_model(player["module"], player["ckpt"], device),
            "module": player["module"],
            "ckpt": player["ckpt"],
        }

    names = [p["name"] for p in players]
    rotations = list(itertools.permutations(names))
    stats = defaultdict(empty_player_stats)
    global_stats = Counter()
    game_records = []

    for game_idx in range(games):
        seat_names = list(rotations[game_idx % len(rotations)])
        policies = [
            Policy(name, loaded[name]["model"], device)
            for name in seat_names
        ]
        arena = Arena(policies, seed + game_idx)
        result = arena.run()
        global_stats["games"] += 1
        global_stats[result["reason"]] += 1
        global_stats["turns"] += result["turns"]

        winner = result["winner"]
        loser = result["loser"]
        shanten = result["shanten_by_seat"]
        if winner is None:
            global_stats["draw"] += 1
        else:
            global_stats["win"] += 1
            winner_name = seat_names[winner]
            stats[winner_name]["wins"] += 1
            if result["reason"] == "tsumo":
                stats[winner_name]["tsumo_wins"] += 1
            elif result["reason"] == "ron":
                stats[winner_name]["ron_wins"] += 1
            elif result["reason"] == "rob_kong":
                stats[winner_name]["rob_kong_wins"] += 1
            stats[winner_name]["seat_wins"][winner] += 1
            if loser is not None:
                stats[seat_names[loser]]["deal_in"] += 1

        for seat, name in enumerate(seat_names):
            s = stats[name]
            s["games"] += 1
            s["turns_sum"] += result["turns"]
            s["seat_games"][seat] += 1
            if winner is None:
                s["draws"] += 1
            final_shanten = shanten[seat]
            s["final_shanten_sum"] += final_shanten
            if seat != winner:
                s["non_win_final_shanten_sum"] += final_shanten
                s["non_win_final_shanten_count"] += 1
            other_shanten = [shanten[i] for i in range(4) if i != seat]
            if final_shanten < min(other_shanten):
                s["best_shanten"] += 1
            if final_shanten <= min(other_shanten):
                s["tied_best_shanten"] += 1
            s["action_counts"].update(policies[seat].action_counts)

        if save_games:
            game_records.append(
                {
                    "game": game_idx,
                    "seed": seed + game_idx,
                    "seat_names": seat_names,
                    "winner": winner,
                    "winner_name": None if winner is None else seat_names[winner],
                    "loser": loser,
                    "loser_name": None if loser is None else seat_names[loser],
                    "reason": result["reason"],
                    "turns": result["turns"],
                    "shanten_by_seat": shanten,
                }
            )
        if progress_every and (game_idx + 1) % progress_every == 0:
            wins = {
                name: stats[name]["wins"]
                for name in names
            }
            print(
                json.dumps(
                    {
                        "progress": game_idx + 1,
                        "games": games,
                        "draws": global_stats["draw"],
                        "wins": wins,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )

    total_games = max(global_stats["games"], 1)
    total_non_draw = global_stats["win"]
    return {
        "config": {
            "games": games,
            "seed": seed,
            "device": device,
            "players": [
                {
                    "name": p["name"],
                    "module": p["module"],
                    "ckpt": p["ckpt"],
                }
                for p in players
            ],
            "rotation_count": len(rotations),
        },
        "global": {
            "games": global_stats["games"],
            "wins": global_stats["win"],
            "draws": global_stats["draw"],
            "draw_rate": global_stats["draw"] / total_games,
            "avg_turns": global_stats["turns"] / total_games,
            "reasons": {
                key: global_stats[key]
                for key in ("tsumo", "ron", "rob_kong", "huang")
            },
        },
        "players": serializable_stats(stats, total_games, total_non_draw),
        "games": game_records if save_games else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--player",
        action="append",
        type=parse_player,
        required=True,
        help="name=model_module:checkpoint; pass exactly four times",
    )
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out", default="")
    parser.add_argument("--save-games", action="store_true")
    parser.add_argument("--progress-every", type=int, default=0)
    args = parser.parse_args()

    if len(args.player) != 4:
        raise SystemExit("--player must be provided exactly four times")
    names = [p["name"] for p in args.player]
    if len(names) != len(set(names)):
        raise SystemExit("player names must be unique")

    torch.set_num_threads(args.threads)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    result = run_fourway(
        args.player,
        args.games,
        args.seed,
        device,
        args.save_games,
        args.progress_every,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")


if __name__ == "__main__":
    main()
