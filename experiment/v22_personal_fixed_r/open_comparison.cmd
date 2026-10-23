@echo off
setlocal
cd /d "%~dp0..\.."
".conda\envs\mano-l20-sim\python.exe" "scripts\view_fixed_r_mujoco_v22.py" --view separate %*
if errorlevel 1 pause
