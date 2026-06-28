#!/usr/bin/env bash
set -euo pipefail

MODEL=AVT
export MODEL
exec "$(dirname "$0")/train_phase.sh" "$@"
