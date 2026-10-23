@echo off
setlocal
cd /d "%~dp0..\.."
".conda\envs\mano-l20-sim\python.exe" "scripts\view_fixed_r_mujoco_v22.py" --motion "%~dp0run1_preserve_points\comparison\old_new_geometry.npz" --output "%~dp0run1_preserve_points\mujoco_comparison" --view separate %*
if errorlevel 1 pause
