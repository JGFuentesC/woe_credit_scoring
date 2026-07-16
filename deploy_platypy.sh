#!/bin/bash
# Deploy woe_credit_scoring to platypy and test everything
# Run from the repo root on your local machine:
#   bash deploy_platypy.sh

set -e
SSH_TARGET="jose@platypy"
PLATYPY_WORKDIR="/home/jose/platypy/woe_credit_scoring"
LOCAL_REPO="$(pwd)"
VENV_DIR="/home/jose/platypy/venvs/woe_credit_scoring"

echo "=== 1. Copy source code ==="
ssh $SSH_TARGET "mkdir -p $PLATYPY_WORKDIR"
rsync -avz \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='graphify-out' \
    --exclude='venv' \
    --exclude='.venv' \
    --exclude='*.egg-info' \
    --exclude='dist' \
    --exclude='reports' \
    ./ $SSH_TARGET:$PLATYPY_WORKDIR/

echo "=== 2. Create venv and install ==="
ssh $SSH_TARGET "bash -s" << 'ENDSSH'
set -e
cd /home/jose/platypy/woe_credit_scoring
python3 -m venv /home/jose/platypy/venvs/woe_credit_scoring
source /home/jose/platypy/venvs/woe_credit_scoring/bin/activate
pip install --upgrade pip -q
pip install -e . -q
pip install fastmcp -q
pip install pytest -q
echo "=== Installation complete ==="
python -c "from woe_credit_scoring import *; print('Imports OK')"
ENDSSH

echo "=== 3. Run tests ==="
ssh $SSH_TARGET "cd $PLATYPY_WORKDIR && source $VENV_DIR/bin/activate && python -m pytest tests/ -v --tb=short" 2>&1 | tail -40

echo "=== 4. Run notebook end-to-end ==="
ssh $SSH_TARGET "cd $PLATYPY_WORKDIR && source $VENV_DIR/bin/activate && jupyter nbconvert --to notebook --execute --inplace 'Usage Example.ipynb' --ExecutePreprocessor.timeout=180" 2>&1

echo "=== 5. Test MCP server starts ==="
ssh $SSH_TARGET "cd $PLATYPY_WORKDIR && source $VENV_DIR/bin/activate && timeout 5 python -m woe_credit_scoring.mcp_server 2>&1 || true"
echo "(Expected: server hangs waiting for stdio — that means it's working)"

echo "=== 6. Test MCP tools directly ==="
ssh $SSH_TARGET "cd $PLATYPY_WORKDIR && source $VENV_DIR/bin/activate && python -c '
import warnings; warnings.filterwarnings(\"ignore\")
from woe_credit_scoring.mcp_server import (
    analyze_dataset, calculate_iv, build_scorecard, score_clients, explain_decision
)
import tempfile, os, pandas as pd

# Test each tool
r = analyze_dataset(\"example_data/train.csv\", \"TARGET\")
assert \"basic_info\" in r, \"analyze_dataset failed\"
print(\"analyze_dataset OK\")

r = calculate_iv(\"example_data/train.csv\", \"TARGET\", max_bins=3)
assert len(r) > 0, \"calculate_iv failed\"
print(f\"calculate_iv OK ({len(r)} features)\")

tmp = tempfile.mktemp(suffix=\".pkl\")
r = build_scorecard(\"example_data/train.csv\", \"TARGET\", model_path=tmp, max_bins=3, iv_threshold=0.1)
assert r[\"status\"] == \"fitted\", \"build_scorecard failed\"
print(f\"build_scorecard OK (AUC={r[\"score_summary\"][\"auc_train\"]:.3f})\")

r = score_clients(tmp, \"example_data/valid.csv\")
assert len(r) > 0 and \"score\" in r[0], \"score_clients failed\"
print(f\"score_clients OK ({len(r)} clients)\")

train = pd.read_csv(\"example_data/train.csv\")
client = {c: train[c].iloc[0].item() if hasattr(train[c].iloc[0], \"item\") and pd.api.types.is_numeric_dtype(train[c]) else str(train[c].iloc[0])
          for c in [col for col in train.columns if col.startswith(\"C_\") or col.startswith(\"D_\")]}
r = explain_decision(tmp, client)
assert any(x[\"feature\"] == \"TOTAL\" for x in r), \"explain_decision failed\"
print(f\"explain_decision OK\")

os.unlink(tmp)
print(\"ALL 5 MCP TOOLS WORKING\")
'"

echo ""
echo "=== DEPLOYMENT COMPLETE ==="
echo "Notebook: http://platypy:8892/lab/tree/woe_credit_scoring/Usage%20Example.ipynb"
echo "MCP server command: python -m woe_credit_scoring.mcp_server"
