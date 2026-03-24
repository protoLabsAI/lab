#!/bin/bash
# Build scaled DPO dataset with category-appropriate teachers
# Usage: nohup bash training/build_scaled_dataset.sh > /mnt/scratch/logs/dataset-build.log 2>&1 &

set -euo pipefail

LAB="$HOME/dev/lab"
VENV="$LAB/.venv/bin/python"
EVALS="$LAB/evals"
WB="$EVALS/results/wildbench"
DATA="$LAB/training/data"

echo "=========================================="
echo "Scaled Dataset Build — $(date)"
echo "=========================================="

# Step 1: Generate Sonnet responses for creative categories (303 tasks, ~$5)
echo ""
echo ">>> Step 1: Generating Sonnet responses for creative tasks..."
cd "$EVALS"
bash run.sh wildbench infer \
    --model claude-sonnet-4-6 \
    --gateway-url http://localhost:4000/v1 \
    --output "$WB/sonnet-creative.json" \
    || echo "Sonnet infer failed"

echo "Sonnet creative responses: $($VENV -c "import json; print(len(json.load(open('$WB/sonnet-creative.json'))))" 2>/dev/null || echo 'failed')"

# Step 2: Download and filter UltraFeedback
echo ""
echo ">>> Step 2: Downloading UltraFeedback creative subset..."
$VENV -c "
import json
from datasets import load_dataset

# Load UltraFeedback — it has 4 model responses per prompt rated on 4 dimensions
ds = load_dataset('openbmb/UltraFeedback', split='train')

creative_keywords = ['story', 'write', 'creative', 'poem', 'narrative', 'character',
                     'fiction', 'dialogue', 'scene', 'roleplay', 'imagine', 'describe',
                     'essay', 'letter', 'blog', 'article', 'rewrite', 'edit', 'improve',
                     'brainstorm', 'ideas', 'suggest', 'advice', 'recommend']

creative_pairs = []
for item in ds:
    instruction = item.get('instruction', '').lower()
    if any(kw in instruction for kw in creative_keywords):
        completions = item.get('completions', [])
        if len(completions) >= 2:
            # Sort by overall_score descending
            scored = [(c, c.get('annotations', {}).get('overall', {}).get('rating', 0)) for c in completions]
            scored.sort(key=lambda x: x[1], reverse=True)

            best = scored[0]
            worst = scored[-1]

            # Only keep if gap >= 2 points
            gap = best[1] - worst[1]
            if gap >= 2 and best[1] >= 4:  # UltraFeedback uses 1-5 scale
                creative_pairs.append({
                    'conversations': [
                        {'from': 'human', 'value': item['instruction']}
                    ],
                    'chosen': {'from': 'gpt', 'value': best[0].get('response', '')},
                    'rejected': {'from': 'gpt', 'value': worst[0].get('response', '')},
                    'source': 'ultrafeedback',
                    'score_gap': gap,
                })

print(f'UltraFeedback creative pairs (gap>=2): {len(creative_pairs)}')
with open('$DATA/ultrafeedback_creative.json', 'w') as f:
    json.dump(creative_pairs, f, indent=2, ensure_ascii=False)
"

echo "UltraFeedback pairs: $($VENV -c "import json; print(len(json.load(open('$DATA/ultrafeedback_creative.json'))))" 2>/dev/null || echo 'failed')"

# Step 3: Build the combined dataset
echo ""
echo ">>> Step 3: Building combined scaled dataset..."
$VENV -c "
import json

# Load all sources
# 1. WildBench: Sonnet chosen (creative) + GPT-5.4-mini chosen (non-creative)
sonnet = json.load(open('$WB/sonnet-creative.json'))
gpt_mini = json.load(open('$WB/gpt-5.4-mini.json'))
base_9b = json.load(open('$WB/9b-base-full.json'))

sonnet_map = {d['session_id']: d for d in sonnet}
gpt_map = {d['session_id']: d for d in gpt_mini}
base_map = {d['session_id']: d for d in base_9b}

creative_ids = set(json.load(open('$DATA/creative_task_ids.json')))

# Build WildBench pairs with category-appropriate teacher
wb_pairs = []
for sid in base_map:
    rejected = base_map[sid]
    rejected_text = rejected.get('output', '')
    if len(rejected_text) < 100:
        continue

    # Pick chosen based on category
    if sid in creative_ids and sid in sonnet_map:
        chosen_text = sonnet_map[sid].get('output', '')
        teacher = 'sonnet'
    elif sid in gpt_map:
        chosen_text = gpt_map[sid].get('output', '')
        teacher = 'gpt-mini'
    else:
        continue

    if len(chosen_text) < 100:
        continue

    messages = rejected.get('conversation_input', [])
    conversations = []
    for msg in messages[:-1]:
        role = 'human' if msg['role'] == 'user' else 'gpt'
        conversations.append({'from': role, 'value': msg['content']})
    conversations.append({'from': 'human', 'value': messages[-1]['content']})

    wb_pairs.append({
        'conversations': conversations,
        'chosen': {'from': 'gpt', 'value': chosen_text},
        'rejected': {'from': 'gpt', 'value': rejected_text},
        'source': f'wildbench_{teacher}',
        'primary_tag': rejected.get('primary_tag', 'unknown'),
    })

# 2. UltraFeedback creative pairs
uf_pairs = json.load(open('$DATA/ultrafeedback_creative.json'))

# Combine
all_pairs = wb_pairs + uf_pairs

# Stats
wb_creative = sum(1 for p in wb_pairs if p.get('primary_tag', '') in
    {'Creative Writing', 'Role playing', 'Brainstorming', 'Editing', 'Advice seeking'})
wb_other = len(wb_pairs) - wb_creative

print(f'WildBench pairs: {len(wb_pairs)} (creative: {wb_creative}, other: {wb_other})')
print(f'UltraFeedback pairs: {len(uf_pairs)}')
print(f'Total combined: {len(all_pairs)}')

with open('$DATA/dpo_scaled_9b.json', 'w') as f:
    json.dump(all_pairs, f, indent=2, ensure_ascii=False)

print(f'Saved to dpo_scaled_9b.json')
"

echo ""
echo "=========================================="
echo "Dataset Build Complete — $(date)"
echo "=========================================="
