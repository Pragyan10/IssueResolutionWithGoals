# This is the readme of the repository.

## 3 Directories:
1. ForIssueSelection
2. ForGoalExtraction
3. ForPatchGeneration


## Files in "ForIssueSelection"
1. select_20_multilang_english.py -> selects 20 issues from SWE Bench Live. Add this inside the git cloned folder SWE Bench live. Set up venv and then setup requirements and run the code. will create a "multilang_gold_selection" directory. 
2. select_20_verified.py -> select 20 issues from swe bench verified that pass the benchmark test. selects at random.


## Files in "ForGoalExtraction"
1. swe_live_selected_20_with_goals.json -> details of the 20 issues selected for swe live. 
2. select_20_swe_verified.json -> details of the 20 issues selected for swe verified.
3. extract_goals_claude_v3.py -> run this code to extract goals for the issues. pass the json without goals and this will add it. 

## Files in "ForPatchGeneration"
1. run_patches.py -> this will run the patch generation. 







