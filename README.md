# Interpretable Machine Learning

This repository collects my coursework from the **Interpretable Machine Learning** course at [**Leibniz University Hannover**](https://www.uni-hannover.de/en/) (Leibniz Universität Hannover, LUH): implementations, tests, and small experiments around model explanations (e.g. partial dependence–style views, Shapley-related ideas, LIME-style tooling, adversarial and counterfactual examples, and neural attribution). It is a personal portfolio and not an official university or course website.

Each `assignment_*` directory is **self-contained** (own `requirements.txt` and `pytest` suite). Use Python **3.9** and a virtual environment per assignment unless you prefer one shared env.

## Quick start

```bash
cd assignment_1   # or any assignment_N folder
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q
```

## Layout

| Folder | Focus (short) |
|--------|----------------|
| `assignment_1` | Notebook-based intro on data and baselines (`exercise.ipynb`, `test_notebook.py`) |
| `assignment_2` | Interpretable structure: trees, linear models, plotting |
| `assignment_3` | ICE / ALE-style curves |
| `assignment_5` | Model-agnostic explanations, cooperative game / Shapley-flavored tasks |
| `assignment_6` | Custom LIME-style explanations |
| `assignment_7` | Adversarial examples and counterfactuals (image + tabular) |
| `assignment_8` | CNN explanations (e.g. Captum-oriented tasks) |

There is no `assignment_4` in this tree; numbering matches the original course set.

## License

Code in this repository is released under the [MIT License](LICENSE), to the extent the author controls the rights in their own solutions. Third-party datasets and assets keep their original terms (see each assignment `README.md`).

