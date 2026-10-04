# This is the readme of the repository.

## 3 Directories:
1. ForIssueSelection
2. ForGoalExtraction
3. ForPatchGeneration

## Files in "ForIssueSelection"
1. select_20_multilang_english.py -> selects 20 issues from SWE Bench Live. Add this inside the git cloned folder SWE Bench live. Set up venv and then setup requirements and run the code. will create a "multilang_gold_selection" directory. 
2. select_20_verified.py -> select 20 issues from swe bench verified that pass the benchmark test. selects at random.

## How to use the files.

## For Verified

> git clone https://github.com/SWE-bench/SWE-bench.git

> cd SWE-bench

> python3 -m venv .venv

> source .venv/bin/activate

> python -m pip install --upgrade pip

> pip install -e .

> pip install datasets

> pip install -e .

> cp ../IssueResolutionWithGoals/ForIssueSelection/select_20_verified.py .

Your directory should then look approximately like:
SWE-bench
- swebench/
- logs/
- select_20_verified.py
- pyproject.toml
- ...

> python select_20_verified.py

## SWE-bench-Live / MultiLang

> cd ..

> git clone https://github.com/microsoft/SWE-bench-Live.git

> cd SWE-bench-Live

> python3 -m venv .venv

> source .venv/bin/activate

> python -m pip install --upgrade pip

> pip install -e .

> pip install datasets

> cp ../IssueResolutionWithGoals/ForIssueSelection/select_20_multilang_english.py .

The directory should now look approximately like:
SWE-bench-Live
- evaluation/
- select_20_multilang_english.py
- pyproject.toml
- ...

> python select_20_multilang_english.py


## Files in "ForGoalExtraction"
1. swe_live_selected_20_with_goals.json -> final file used | details of the 20 issues selected for swe live. 
2. swe_verified_selected_20_with_goals.json -> final file used | details of the 20 issues selected for swe verified.
3. selected_20_verified.json -> this is the selected 20 issues for goal extraction.
4. selected_20_live.json -> this is the selected 20 issues for goal extraction. 
5. extract_goals_claude_v3.py -> run this code to extract goals for the issues. pass the json without goals and this will add it. 

## How to use the files.

Write the command in your terminal. 

> ls

This should show: 

goal_extraction
- extract_goals_claude_v3.py
- selected_20_verified.json
- selected_20_live.json
- swe_verified_selected_20_with_goals.json
- swe_live_selected_20_with_goals.json

> python3 --version

python version should be more than 3.9 

> python3 -m venv .venv

> source .venv/bin/activate

> pip install --upgrade pip

> pip install anthropic

> python -c "import anthropic; print(anthropic.__version__)"

Before running the following command make sure to have an API key from Claude Platform. 

> export ANTHROPIC_API_KEY="sk-ant-...your key..."

Lets test with just 3 instances for now. 

> python extract_goals_claude_v3.py selected_20_verified.json --limit 3

Full run 

> python extract_goals_claude_v3.py selected_20_verified.json selected_20_live.json


## Files in "ForPatchGeneration"
1. run_patches.py -> this will run the patch generation. 
2. swe_live_selected_20_with_goals.json -> details of the 20 issues selected for swe live. 
3. swe_verified_selected_20_with_goals.json -> details of the 20 issues selected for swe verified.

## How to use the files.

Write the command in your terminal. 

> ls
patch_gen
- run_patches.py
- swe_verified_selected_20_with_goals.json
- swe_live_selected_20_with_goals.json

> python3 -m venv .venv

> source .venv/bin/activate     

> pip install --upgrade pip

> pip install "mini-swe-agent==2.4.6"

> export OPENAI_API_KEY="sk-..."

Check the Docker images (no model calls, no cost)

> python run_patches.py swe_verified_selected_20_with_goals.json swe_live_selected_20_with_goals.json --check-images

Look at the prompts (no cost)

>python run_patches.py swe_verified_selected_20_with_goals.json swe_live_selected_20_with_goals.json --conditions baseline goal --dry-run

> cat patch_runs/verified/goal/rep_1/django__django-12143/prompt.txt

Small test: one issue, both conditions, one repetition

> python run_patches.py swe_verified_selected_20_with_goals.json --conditions baseline goal --reps 1 --instance-ids django__django-14493

Full run   

> python run_patches.py swe_verified_selected_20_with_goals.json swe_live_selected_20_with_goals.json --conditions baseline goal --reps 3 --workers 2
