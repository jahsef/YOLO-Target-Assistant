@echo off
REM Manual test runner. No CI on purpose -- these need a CUDA device and the local
REM (gitignored) engine files, so they only mean anything on your box.
REM
REM   run_tests.bat                     correctness only, fast
REM   run_tests.bat perf                correctness + perf, records tests\results\
REM   run_tests.bat perf --update-baseline    ...and adopt these numbers as the baseline
REM
REM Structured numbers land in tests\results\latest.md + latest.json (both committed,
REM so regressions are visible across branches). Raw console output goes to
REM tests\results\console.txt, which is gitignored -- it's just for eyeballing.

setlocal
cd /d "%~dp0"
if not exist tests\results mkdir tests\results
set OUT=tests\results\console.txt

echo === correctness (%DATE% %TIME%) === > "%OUT%"
python -m pytest tests -q >> "%OUT%" 2>&1
set CORRECTNESS=%ERRORLEVEL%
type "%OUT%"

if /I not "%1"=="perf" goto :done

echo. >> "%OUT%"
echo === perf (%DATE% %TIME%) === >> "%OUT%"
python -m pytest -m perf -q %2 %3 %4 >> "%OUT%" 2>&1
set PERFRESULT=%ERRORLEVEL%
echo.
powershell -NoProfile -Command "Get-Content '%OUT%' | Select-Object -Skip 1 | Select-String -Pattern '=== perf' -Context 0,10000 | Select-Object -First 1" 2>nul
type tests\results\latest.md

if not "%PERFRESULT%"=="0" (
    echo.
    echo PERF SUITE FAILED ^(a measurement errored -- regressions never fail^)
    exit /b %PERFRESULT%
)

:done
if not "%CORRECTNESS%"=="0" (
    echo.
    echo CORRECTNESS FAILURES -- see %OUT%
    exit /b %CORRECTNESS%
)
echo.
echo all good
exit /b 0
