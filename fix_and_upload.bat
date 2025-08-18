@echo off
cd /d "d:\NewProjects\LowCostChattingBot\npc-graph-chat"
echo === Resolving Git Issues ===
echo.
echo === Step 1: Abort current rebase ===
git rebase --abort
echo.
echo === Step 2: Check current status ===
git status --short
echo.
echo === Step 3: Pull latest changes ===
git pull origin main --no-edit
echo.
echo === Step 4: Add and commit all files ===
git add -A
git commit -m "feat: complete ChatTogether project with all versions and documentation"
echo.
echo === Step 5: Push to GitHub ===
git push origin main
echo.
echo === Upload Complete! ===
echo Your project is now available at:
echo https://github.com/RanMo1990/ChatTogether-with-Neo4j-Memory-and-Lightweight-RAG
echo.
pause
