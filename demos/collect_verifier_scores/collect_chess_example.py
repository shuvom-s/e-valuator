from datasets import load_dataset
import chess
import chess.pgn
import chess.engine
import io
import shutil
import pandas as pd
from tqdm import tqdm
import numpy as np
import sys

HF_DATASET = "patrickfrank1/chess-pgn-games"  # full PGN games from lichess :contentReference[oaicite:2]{index=2}
MAX_GAMES = 200       # change as desired (full dataset is huge)
ENGINE_PATH = shutil.which("stockfish") or "/usr/local/bin/stockfish"
ENGINE_DEPTH = 12    # lower for speed, higher for strength



save_path = sys.argv[1] if len(sys.argv) > 1 else "chess_verifier_scores.csv"

## note: install stockfish here: https://stockfishchess.org/download/
if ENGINE_PATH is None:
    raise RuntimeError(
        "Could not find Stockfish in PATH. Install it (e.g. `brew install stockfish`) "
        "or hard-code ENGINE_PATH to the engine binary."
    )

def result_to_winner(result_str: str):
    """
    PGN Result -> {'W','B','T'} or None
    """
    if result_str == "1-0":
        return "W"
    elif result_str == "0-1":
        return "B"
    elif result_str in ("1/2-1/2", "½-½"):
        return "T"
    else:
        return None

# Load PGN games from HuggingFace
print("Loading games from HuggingFace...")
ds = load_dataset(HF_DATASET, split="train")

rows = []

print(f"Using Stockfish at: {ENGINE_PATH}")
with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as engine:
    limit = chess.engine.Limit(depth=ENGINE_DEPTH)

    for game_idx in tqdm(range(min(MAX_GAMES, len(ds))), desc="Games"):
        pgn_text = ds[game_idx]["text"]
        if not pgn_text.strip():
            continue

        # Parse PGN
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            continue

        result_str = game.headers.get("Result", "*")
        winner = result_to_winner(result_str)

        # skip games without a clear result
        if winner is None:
            continue

        board = game.board()
        move_num = 0  

        for move in game.mainline_moves():
            board.push(move)
            move_num += 1

            # Ask engine for evaluation of current position
            info = engine.analyse(board, limit=limit)

            # score from White's POV; mate positions mapped to a large score
            score_cp = info["score"].white().score(mate_score=100000)

            rows.append(
                {
                    "game_idx": game_idx,
                    "move_num": move_num,       
                    "bot_score": score_cp,      # centipawns
                    "winner": winner,          
                }
            )

# Build DataFrame
df = pd.DataFrame(rows)

df["game_idx"] = df["game_idx"].apply(lambda x: f"game_{x}")

df["uq_problem_idx"] = df["game_idx"]
df['stockfish_probability'] = (50 + 50 * (2 / (1 + np.exp(-0.00368208 * df['bot_score'])) - 1))/100
df["judge_probability"] = df["bot_score"]
df["solved"] = df["winner"] == "W"
df["num_steps"] = df["move_num"]

df.to_csv(save_path, index=False)