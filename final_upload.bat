@echo off
cd /d "d:\NewProjects\LowCostChattingBot\npc-graph-chat"
echo === Final Upload Solution ===
echo.
echo === Step 1: Check what we have locally ===
git log --oneline -3
echo.
echo === Step 2: Force push to overwrite remote ===
echo WARNING: This will overwrite the GitHub repository with your local files
set /p confirm="Continue? (y/N): "
if /i "%confirm%" NEQ "y" (
    echo Upload cancelled.
    pause
    exit /b
)
echo.
echo === Step 3: Adding all files ===
git add -A
git status --short
echo.
echo === Step 4: Commit any remaining changes ===
git commit -m "feat: final upload of complete ChatTogether project" || echo "Nothing to commit"
echo.
echo === Step 5: Force push to GitHub ===
git push origin main --force
echo.
echo === SUCCESS! ===
echo All your files are now uploaded to GitHub at:
echo https://github.com/RanMo1990/ChatTogether-with-Neo4j-Memory-and-Lightweight-RAG
echo.
echo Your project includes:
echo - 4 versions of the chatbot (1.0 to 4.0)
echo - Complete documentation (README files)
echo - Configuration and utility files
echo - Safe .gitignore to protect credentials
echo.
pause
