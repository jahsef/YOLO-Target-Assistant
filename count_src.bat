@echo off
setlocal enabledelayedexpansion

set total=0

echo Counting lines of code in src/
echo.

REM Count Python files in src/
for /r src/ %%f in (*.py) do (
    set count=0
    for /f "usebackq tokens=*" %%a in ("%%f") do (
        set "line=%%a"
        REM Remove leading/trailing spaces
        set "line=!line: =!"
        REM Skip empty lines and lines starting with #
        if not "!line!"=="" (
            if not "!line:~0,1!"=="#" (
                set /a count+=1
            )
        )
    )
    set /a total+=!count!
    echo %%f: !count! lines
)


echo.
echo ========================================
echo Total lines of code: !total!
echo ========================================
pause
