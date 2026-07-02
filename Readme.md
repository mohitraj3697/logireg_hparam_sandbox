# logireg-hparam-sandbox

A small Streamlit app to play around with Logistic Regression hyperparameters and actually *see* what they do to the decision boundary, instead of just reading about them.

I built this mostly because I kept forgetting how `C`, `penalty`, and `solver` interact with each other, and reading the sklearn docs wasn't sticking. Watching the boundary shift in real time made it click a lot faster.

## What it does

- Pick between a binary or multiclass toy dataset (generated with `make_blobs`)
- Tune the usual Logistic Regression knobs from the sidebar:
  - `penalty` (l1, l2, elasticnet, none)
  - `C`
  - `solver`
  - `max_iter`
  - `multi_class`
  - `l1_ratio` (only matters if you're using elasticnet)
- Hit "Run Algorithm" and it fits the model, draws the decision boundary using a meshgrid, and shows you the test accuracy

Nothing fancy — no saving models, no uploading your own data (yet). Just a fast way to build intuition.

## Running it locally

```bash
git clone https://github.com/mohitraj3697/logireg_hparam_sandbox.git
cd logireg-hparam-sandbox
pip install -r requirements.txt
streamlit run app.py
```

It'll open in your browser at `localhost:8501`.

## Known issues / things I still need to fix

- The accuracy label currently says "Decision Tree" — copy-paste leftover from another project, not accurate here, need to change it to Logistic Regression
- `l1_ratio` is being cast to `int`, so it only ever ends up as 0 or 1. Should be a float between 0 and 1 for elasticnet to actually work right
- Not every solver supports every penalty (e.g. `liblinear` doesn't support `none`), and right now the app doesn't guard against that — it'll just throw an sklearn error if you pick an invalid combo. Might add validation later.

## Why this exists

Mostly a learning project — I wanted to actually understand how regularization strength and solver choice change the decision boundary, not just memorize which solver supports which penalty. If it's useful to anyone else poking around with logistic regression, cool.

## License

MIT — do whatever you want with it.