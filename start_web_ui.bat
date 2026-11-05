@echo off
cd /d "%~dp0"
"C:\Users\zyx20\anaconda3\envs\myml\python.exe" -m tools.web.server --port 8811
pause
