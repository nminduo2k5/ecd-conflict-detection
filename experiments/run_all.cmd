@echo off
rem Re-run every experiment into a separate results folder (ecd\results_rerun by default).
rem Runs run_all.ps1 with a per-process execution-policy bypass, so no system setting changes.
rem   experiments\run_all.cmd                        -> all steps
rem   experiments\run_all.cmd -FreshCache            -> also recompute Z3 ground truth + labels
rem   experiments\run_all.cmd -Steps rq2,rq2_complex -> a subset of steps
rem   experiments\run_all.cmd -ResultsDir D:\runs\ecd_2026-10-28
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_all.ps1" %*
exit /b %ERRORLEVEL%
