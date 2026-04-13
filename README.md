# FEATURE_SELECTION

Autonomous tabular feature engineering pipeline for the **Code Risk** hackathon.

## What it does

The project reads relational tabular data from `data/`, understands schema from `readme.txt`, builds deterministic joins, generates feature ideas with an LLM, turns some of them into executable pandas feature builders, combines them with legacy baseline features, trains CatBoost, selects the **top 5 numeric features**, and saves final outputs to `output/`.

Main entrypoint:

```bash
python run.py
```

## Current pipeline

1. **Input ingestion**  
   `InputRepository` reads `train.csv`, `test.csv`, additional tables, and `readme.txt`, then builds an `InputBundle` with structured metadata.

2. **README enrichment**  
   `ReadmeService` uses the LLM once to enrich table and column descriptions.

3. **Deterministic join planning**  
   `JoinPlanner` builds train-centric join candidates and a join plan without using the LLM.

4. **Feature idea generation**  
   The LLM proposes a small list of high-level feature ideas based on schema and available joins.

5. **Feature code generation**  
   For each idea, the LLM generates a short pandas function:
   `build_feature(train_df, tables) -> pd.DataFrame`

6. **Guarded feature execution**  
   Generated code is validated and executed safely. Successful feature frames are merged into the prepared train/test datasets.

7. **Legacy executor fallback/features**  
   The older deterministic feature executor is applied on top of the current prepared datasets to add extra baseline features.

8. **CatBoost training and top-5 selection**  
   The pipeline trains CatBoost on numeric features, ranks them by importance, keeps the top 5, and saves final outputs.

## Outputs

After a successful run, the project saves:

- `output/train.csv`
- `output/test.csv`

These files contain:

- original raw columns from source `train/test`
- the selected top-5 engineered features

## Design principles

- **Deterministic where possible**
  - ingestion
  - join planning
  - feature merging
  - baseline legacy execution
  - CatBoost training and selection

- **LLM only where semantics help**
  - README understanding
  - feature idea generation
  - pandas feature code generation

- **Tolerant execution**
  - if some generated features fail, the pipeline continues with the successful ones

## Current status

The project already works end-to-end:

- loads real data
- enriches schema
- builds joins
- generates feature ideas
- generates and applies some pandas-based features
- adds legacy baseline features
- trains CatBoost
- selects top 5 features
- saves final train/test outputs

## Known limitations

- not every generated feature works successfully
- code generation is still prompt-sensitive
- final scoring currently uses only numeric features
- there is no full retry/refinement loop yet
- feature deduplication is still minimal

## Repository layout

```text
src/
  agent/
    graph.py
    runtime.py
    state.py
    nodes/
  common/
    io/
    readme/
    joins/
    feature_ideas_generation/
    feature_codegen/
    stats/
```

## Goal

The current goal is practical, not perfect:

build a reproducible, debuggable, end-to-end baseline that produces useful engineered features and final submission-ready datasets.
