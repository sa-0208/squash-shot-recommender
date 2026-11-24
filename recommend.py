"""Rank possible shots for a supplied 2D court state.

Example:
    python recommend.py --hitter You --hx D --hy 4 --ox E --oy 6 --bx D --by 3

This is a prototype ranking tool. Its probabilities are trained on fictional
examples and must not be interpreted as real squash win probabilities.
"""

import argparse
import pandas as pd

from src.squash_model import load_saved_model, prepare_features

CANDIDATES = [
    ("Drive", "Forehand", "Straight", "", "No", "Fast", "D", 7),
    ("Drive", "Backhand", "Cross-court", "", "No", "Fast", "G", 7),
    ("Drop", "Forehand", "Straight", "", "No", "Slow", "D", 2),
    ("Drop", "Backhand", "Cross-court", "", "No", "Slow", "G", 2),
    ("Kill", "Forehand", "Straight", "", "No", "Fast", "D", 1),
    ("Kill", "Backhand", "Cross-court", "", "No", "Fast", "G", 1),
    ("Boast", "Forehand", "Straight", "2", "No", "Medium", "B", 4),
    ("Boast", "Backhand", "Straight", "3", "No", "Medium", "G", 4),
    ("Trickle boast", "Forehand", "Straight", "2", "No", "Slow", "B", 2),
    ("Lob", "Backhand", "Cross-court", "", "No", "Medium", "G", 7),
    ("Volley drive", "Forehand", "Straight", "", "Yes", "Fast", "D", 7),
    ("Volley drop", "Backhand", "Cross-court", "", "Yes", "Slow", "G", 2),
    ("Cross-court nick", "Forehand", "Cross-court", "", "No", "Fast", "H", 1),
    ("Straight nick", "Backhand", "Straight", "", "No", "Fast", "D", 1),
    ("Reverse boast", "Forehand", "Cross-court", "3", "No", "Medium", "B", 4),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Rank squash shot options.")
    parser.add_argument("--hitter", choices=["You", "Opponent"], default="You")
    parser.add_argument("--hx", choices=list("ABCDEFGH"), default="D")
    parser.add_argument("--hy", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--ox", choices=list("ABCDEFGH"), default="E")
    parser.add_argument("--oy", type=int, choices=range(1, 9), default=6)
    parser.add_argument("--bx", choices=list("ABCDEFGH"), default="D")
    parser.add_argument("--by", type=int, choices=range(1, 9), default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_saved_model()

    rows = []
    for shot, hand, direction, walls, volley, speed, tx, ty in CANDIDATES:
        rows.append({
            "Hitter": args.hitter,
            "Hitter position X (A-H)": args.hx,
            "Hitter position Y (1-8)": args.hy,
            "Opponent position X (A-H)": args.ox,
            "Opponent position Y (1-8)": args.oy,
            "Ball position X (A-H)": args.bx,
            "Ball position Y (1-8)": args.by,
            "Shot": shot,
            "Hand": hand,
            "Direction": direction,
            "Boast wall count": walls,
            "Volley?": volley,
            "Target / landing X (A-H)": tx,
            "Target / landing Y (1-8)": ty,
            "Ball speed": speed,
        })

    candidates = pd.DataFrame(rows)
    features = prepare_features(candidates)
    candidates["Model score (not calibrated)"] = model.predict_proba(features)[:, 1]
    ranked = candidates.sort_values(
        "Model score (not calibrated)", ascending=False
    )

    display_cols = [
        "Shot", "Hand", "Direction", "Boast wall count", "Volley?",
        "Target / landing X (A-H)", "Target / landing Y (1-8)",
        "Ball speed", "Model score (not calibrated)",
    ]
    print("\nRanked candidate shots:")
    print(ranked[display_cols].to_string(
        index=False,
        formatters={"Model score (not calibrated)": "{:.3f}".format}
    ))
    print("\nImportant: scores are model rankings, not trustworthy real-world")
    print("probabilities. The training examples are invented.")


if __name__ == "__main__":
    main()
