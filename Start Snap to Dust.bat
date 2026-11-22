@echo off
title Snap to Dust
cd /d "%~dp0"
python snap_dust.py %*
if errorlevel 1 pause
