@echo off
setlocal
set "PATH=C:\Program Files\MiKTeX\miktex\bin\x64;%PATH%"

echo ============================================================
echo  Compiling: NEV NEG Model Brief Paper (XeLaTeX)
echo ============================================================

echo [1/4] xelatex (first pass)...
xelatex -synctex=1 -interaction=nonstopmode main.tex
if %ERRORLEVEL% neq 0 (
    echo ERROR: xelatex first pass failed
    exit /b 1
)

echo [2/4] bibtex...
bibtex main
if %ERRORLEVEL% neq 0 (
    echo WARNING: bibtex had issues (check references)
)

echo [3/4] xelatex (second pass)...
xelatex -synctex=1 -interaction=nonstopmode main.tex
if %ERRORLEVEL% neq 0 (
    echo ERROR: xelatex second pass failed
    exit /b 1
)

echo [4/4] xelatex (third pass)...
xelatex -synctex=1 -interaction=nonstopmode main.tex
if %ERRORLEVEL% neq 0 (
    echo ERROR: xelatex third pass failed
    exit /b 1
)

echo.
echo ============================================================
echo  Compilation successful! Output: main.pdf
echo ============================================================
endlocal
