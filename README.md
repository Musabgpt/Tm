# MusabAI Samsung Phone Agent

Tiny Arabic intent model for Samsung/Android phone commands.

## Goal
Arabic speech/text → intent + parameters → Android/Termux executor.

Example:

`ابعت لمحمد مرحبا` → `SEND_MESSAGE(contact=محمد, text=مرحبا)`

The model is intentionally tiny. It does not execute Android commands itself; an external validated executor performs the action.

## Kaggle

The training notebook is in `notebooks/MusabAI_Samsung_Phone_Agent_Kaggle.ipynb`.

It:
- starts from `asafaya/bert-mini-arabic` (~11.6M parameters)
- discovers Samsung/Galaxy/Android datasets in `/kaggle/input`
- builds Arabic phone-command training data
- fine-tunes intent classification
- saves resumable checkpoints
- exports the trained model and command schema

The target is a cumulative 30-hour training budget across resumable Kaggle sessions, not one 30-hour session.

## Safety

Model output must pass a strict allow-list/parameter validator before Android execution. Never turn free-form model text directly into shell commands.
