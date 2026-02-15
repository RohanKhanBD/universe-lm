---
description: A complete automated pipeline for planning, implementing, and verifying research ideas in code.
---

# Implementation and Testing Workflow

// turbo-all

Follow these steps to safely implement a research idea and verify its effectiveness:

1.  **Baseline Stat Verification**:
    - Identify the current baseline metrics (e.g., from `plots/metrics_8000000_*.json`).
    - Record the `val_loss` (baseline targeted is approx 4.9413 for 8M tokens).

2.  **Implementation Planning**:
    - Use the `experiment-planner` skill to define the changes.
    - Focus on adding configuration flags to `Configs` and conditional logic in the relevant modules (e.g., `optimizers/muon.py`).

3.  **Code Implementation**:
    - Apply the changes to the codebase.
    - Ensure all new parameters are exposed in the training scripts.

4.  **Implementation Review**:
    - Use the `implementation-reviewer` skill to check the code for bugs.
    - Verify that the baseline path is untouched when the new flag is False.

5.  **Baseline Regression Test**:
    - Run the command: `python train_llm.py --train_tokens 8000000` (ensure any new experimental flags are OFF).
    - Use the `experiment-runner` skill to compare the results with the recorded baseline.
    - If there is a regression, fix the code and repeat this step.

6.  **Experiment Execution**:
    - Run the command: `python train_llm.py --train_tokens 8000000 [NEW_EXPERIMENTAL_FLAGS]`
    - Wait for completion and capture the results.

7.  **Iterative Research Loop (The "Circle")**:
    - **Step A: Deep Analysis**: 
        - Immediately use `experiment-analyzer` on the run. 
        - Determine if the failure was due to math (theory), overhead (system), or noise (stochastic).
    - **Step B: Theory & Review**:
        - Use `idea-revisor` to integrate the "Lessons Learned" into a new **V-Next** proposal.
        - Ensure the new math directly addresses the failure mode of the previous run.
    - **Step C: Code Implementation**:
        - Use `experiment-planner` to update the implementation.
    - **Step D: Verification**:
        - Run the **Baseline Regression Test** again to ensure no "bit-rot."
        - Run the **V-Next Experiment**.
    - **Loop Verification**: If `Experiment_Metric <= Baseline_Metric`, GOTO Step A.

8.  **Final Victory Phase**:
    - Once the experiment beats the baseline:
        - Update the `docs/research/idea_log.md` with the "Validated" status.
        - Update the research paper with the Final Results and the "Path to Success" narrative.
        - Save the final optimized model checkpoints.

