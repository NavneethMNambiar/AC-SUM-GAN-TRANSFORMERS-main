@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

REM Change these variables 
REM set base_path=<path_to_experiment>
REM set exp_name="sigma0.1"
REM set dataset_name="TVSum"
REM set eval_method='avg'

REM OR use arguments (example usage: evaluate_exp.bat <path_to_experiment> sigma0.1 TVSum avg)
set base_path=%1
set exp_name=%2
set dataset_name=%3
set eval_method=%4

set "exp_path=%base_path%\%dataset_name%\%exp_name%"  

for /l %%i in (0,1,4) do (
    set "path=!exp_path!\logs\split%%i"
    C:\Users\jglad\anaconda3\envs\tf\python.exe exportTensorFlowLog.py !path! !path!
    set "results_path=!exp_path!\results\split%%i"
    C:\Users\jglad\anaconda3\envs\tf\python.exe compute_fscores.py !results_path! %dataset_name% %eval_method%
)

ENDLOCAL
