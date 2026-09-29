# MusabAI Samsung Phone Agent

Tiny Arabic NLU model for Samsung/Android phone commands.

## Goal
Arabic speech/text → intent + parameters → Android/Termux executor.

Example:

`ابعت لمحمد مرحبا` → `SEND_MESSAGE(contact=محمد, text=مرحبا)`

The model is intentionally tiny. It does not execute Android commands itself; an external validated executor performs the action.

## Training: CPU only

This project does **not** use Kaggle or a GPU.

Training runs from GitHub Actions on CPU and is resumable through checkpoints/artifacts. The workflow is `.github/workflows/train-cpu.yml` and can be started manually from the **Actions** tab.

The trainer starts from `asafaya/bert-mini-arabic` (~11.6M parameters), generates Arabic phone-command examples, fine-tunes intent classification, evaluates the model, and exports the model plus command schema.

Because CPU training is slow, each GitHub Actions run is bounded to a few hours. Repeated runs continue from the latest checkpoint rather than pretending a single run can last 30 hours.

## Data

`data/commands_seed.jsonl` contains the initial Arabic/Samsung phone-command seed set. The training script expands it with controlled Arabic and Levantine-style wording variants.

## Safety

Model output must pass a strict allow-list/parameter validator before Android execution. Never turn free-form model text directly into shell commands.
