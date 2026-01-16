# Run Complete Experiment Pipeline

Write-Host "=== Step 1: Generating Expert Traces ==="
python -m src.data_gen.generate_traces
if ($LASTEXITCODE -ne 0) { Write-Error "Step 1 Failed"; exit $LASTEXITCODE }

Write-Host "=== Step 2: Creating Preference Pairs (DPO Dataset) ==="
python src/data_gen/create_preference_pairs.py
if ($LASTEXITCODE -ne 0) { Write-Error "Step 2 Failed"; exit $LASTEXITCODE }

Write-Host "=== Step 3: Running DPO Training (Fine-tuning Phi-2) ==="
# Using parameters for a quick run. Adjust epochs/batch_size for full training.
python src/llm/train_dpo.py --epochs 1 --load_in_4bit --output_dir results_dpo
if ($LASTEXITCODE -ne 0) { Write-Error "Step 3 Failed"; exit $LASTEXITCODE }

Write-Host "=== Step 4: Evaluating the Fine-tuned Model ==="
# Evaluate the base model first (optional, for comparison)
# python src/experiments/evaluate_pipeline.py --model_name microsoft/phi-2 --num_episodes 10

# Evaluate the DPO-trained adapter
python src/experiments/evaluate_pipeline.py --model_name microsoft/phi-2 --adapter_path results_dpo --num_episodes 20
if ($LASTEXITCODE -ne 0) { Write-Error "Step 4 Failed"; exit $LASTEXITCODE }

Write-Host "=== Experiment Pipeline Completed Successfully! ==="
