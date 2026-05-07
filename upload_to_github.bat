@echo off
echo ========================================
echo Git Upload Script for Turbulence Balloon Simulator
echo ========================================
echo.

cd /d "L:\v3\AI foundation\turbulence_balloon_simulator"

echo Step 1: Initialize git repository...
git init
if errorlevel 1 (
    echo Git init failed or already initialized
)

echo.
echo Step 2: Add remote repository...
git remote remove origin 2>nul
git remote add origin https://github.com/aaachill886/turbulence_balloon_simulator.git
if errorlevel 1 (
    echo Failed to add remote
    pause
    exit /b 1
)

echo.
echo Step 3: Add all files...
git add .
if errorlevel 1 (
    echo Failed to add files
    pause
    exit /b 1
)

echo.
echo Step 4: Create commit...
git commit -m "Complete turbulence balloon simulator with adaptive fog visualization"
if errorlevel 1 (
    echo Commit failed - maybe no changes or already committed
)

echo.
echo Step 5: Push to GitHub...
echo You may need to authenticate with GitHub
git branch -M main
git push -u origin main --force
if errorlevel 1 (
    echo Push failed - check your GitHub credentials
    pause
    exit /b 1
)

echo.
echo ========================================
echo Upload completed successfully!
echo ========================================
pause
