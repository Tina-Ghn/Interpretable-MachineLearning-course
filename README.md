# Interpretable Machine Learning — portfolio coursework

This repository collects coursework for an **Interpretable Machine Learning** course: implementations, tests, and small experiments around model explanations (e.g. partial dependence–style views, Shapley-related ideas, LIME-style tooling, adversarial and counterfactual examples, and neural attribution).

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

## What was changed for a public portfolio

- **Official `exercise.pdf` handouts** are not included here, to respect instructor/course copyright. Your own write-ups or links to public materials belong in your CV or a blog post, not necessarily in this repo.
- **GitHub Classroom** automation (badges, Classroom workflows, autograding JSON) was removed so the project reads as a normal open repository and CI runs plain `pytest`.
- **Student identity files** (`user_info.txt`) must never be committed; they are listed in `.gitignore`.

If you are a current student elsewhere, do not copy solutions—use the course’s own rules.

## License

Code in this repository is released under the [MIT License](LICENSE), to the extent the author controls the rights in their own solutions. Third-party datasets and assets keep their original terms (see each assignment `README.md`).
