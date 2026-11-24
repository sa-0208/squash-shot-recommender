# Squash Shot Recommender (prototype)

A small Python project that:
1. reads the 30 synthetic rallies,
2. converts court coordinates to numbers,
3. trains a shot-outcome classifier,
4. evaluates it while keeping whole rallies separate, and
5. ranks a menu of candidate shots for a supplied court state.

## Important limitation

The included 30 rallies are fictional and approximate. This project is a software
prototype: it tests the data pipeline and model workflow, **not** whether the
system can predict real squash outcomes. Do not describe its scores as real
performance. Replace or supplement the synthetic examples with consistently
annotated real-match data before drawing sporting conclusions.

This is a first supervised-learning baseline, not yet a full Monte Carlo Tree
Search (MCTS) engine. The current spreadsheet does not contain a reliable model
of how the ball and players transition after every candidate action, or how an
opponent chooses a response. MCTS needs those transition/response models; adding
a search tree now would create false precision.

## Court coordinates

- `A1` is the bottom-left square.
- Letters A to H increase from left to right.
- Numbers 1 to 8 increase from bottom to top.
- Therefore A1 = bottom-left, H1 = bottom-right, A8 = top-left, H8 = top-right.

## Project structure

```text
squash-shot-recommender/
├── data/
│   ├── squash_synthetic_30_rallies.xlsx
│   └── squash_synthetic_30_rallies.csv
├── src/
│   ├── __init__.py
│   └── squash_model.py
├── artifacts/                 # created when you train; do not need to commit
├── train_model.py
├── recommend.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

Install Python 3.10 or newer. In a terminal, from this project folder:

### Windows

```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Train the model

```bash
python train_model.py
```

This creates:
- `artifacts/squash_shot_model.joblib` — trained model
- `artifacts/evaluation_metrics.json` — evaluation summary

The evaluation split is grouped by `Rally ID`, so shots from the same rally
cannot be in both the training and test sets. This reduces a common form of
data leakage.

The model predicts `Direct winning shot?`. It deliberately does **not** use
`Shot result`, `Point-ending shot?`, `Point-ending event`, `Rally winner`, or
`Point-ending shot number` as input features, because those reveal the outcome.

## Rank candidate shots

After training, try:

```bash
python recommend.py --hitter You --hx D --hy 4 --ox E --oy 6 --bx D --by 3
```

The coordinate flags are:
- `--hx`, `--hy`: hitter position
- `--ox`, `--oy`: opponent position
- `--bx`, `--by`: ball position
- `--hitter`: who is hitting (`You` or `Opponent`)

The program ranks a fixed candidate menu. The menu can be expanded in
`recommend.py`. The scores are not calibrated probabilities and are not
meaningful as real-world estimates while trained on synthetic data.

## Put it on GitHub

1. Create a new empty repository on GitHub, e.g. `squash-shot-recommender`.
2. Put the files/folders above inside a local folder with that name.
3. Open a terminal in that folder and run:

```bash
git init
git add .
git commit -m "Build squash shot recommender prototype"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/squash-shot-recommender.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username.

Do not commit your `.venv` folder. The `.gitignore` file excludes it and the
generated `artifacts/` folder. Keep the data workbook/CSV in the repository
while it is small and contains no private information.

## Improving the dataset

For every shot, record the state immediately before contact and the shot
actually played. Keep one row per shot. Keep the point-ending labels only on
the final shot row. When you begin collecting real rallies, add a `Data type`
value of `Real` and preserve the distinction between real and synthetic data.

For a credible evaluation:
- collect many more rallies from multiple matches and players;
- record positions consistently, with uncertain values left blank;
- split training and testing by match (not merely by individual shot);
- evaluate on matches that were never used for training or tuning;
- compare against simple baselines and inspect errors.

## Next research stage: MCTS

A genuine MCTS version should follow once there is enough real data to estimate:
- shot execution success / error probability for a given state and shot;
- where the ball and both players move after a shot;
- the opponent's likely response, including a strong-response policy;
- how to score a state and terminal rally outcomes.

Then MCTS can search alternating player decisions over simulated future shots.
The current baseline is the first component, not the finished game-theory model.
