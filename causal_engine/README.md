# Causal Engine

A complete, production-ready causal-reasoning system using modern causal ML methods.

## Features

- **Structure Learning**: Learn causal graphs from data using PC, GES, or NOTEARS.
- **Structural Causal Models (SCM)**: Neural network-based SCMs for non-linear relationships.
- **Interventions**: Perform `do()` operator interventions.
- **Counterfactuals**: Estimate counterfactual outcomes.
- **Generative Models**: CausalVAE for latent factor modeling.
- **API**: FastAPI microservice for inference and interventions.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Learn Graph
```bash
python -m src.causal_graph.structure_learning
```

### 2. Train Causal Engine
```bash
python -m src.training.train_causal
```

### 3. Train Baseline
```bash
python -m src.training.train_baseline
```

### 4. Evaluate
```bash
python -m src.evaluation.evaluate
```

### 5. Launch API
```bash
uvicorn src.service.server:app --reload
```

## Docker

```bash
docker-compose up --build
```

