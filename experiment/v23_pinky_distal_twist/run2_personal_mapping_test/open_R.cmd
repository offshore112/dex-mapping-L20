@echo off
setlocal
cd /d "%~dp0..\..\.."
".conda\envs\mano-l20-sim\python.exe" "scripts\view_fixed_r_mujoco_v22.py" --motion "%~dp0fixed_r\motion_geometry.npz" --output "%~dp0mujoco" --view r-only --label "v23 repeat" %*
if errorlevel 1 pause
