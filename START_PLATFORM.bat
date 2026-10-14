@echo off
rem Starts the AI CRM Procurement Platform on this machine and opens it in your browser.
rem Safe default: your Supabase CRM (read-only) + SIMULATED research (no Gemini/Tavily quota used, nothing written).
rem For real research see the header of serve_platform.py.
cd /d "%~dp0"
".venv-agentic\Scripts\python.exe" serve_platform.py --open
pause
