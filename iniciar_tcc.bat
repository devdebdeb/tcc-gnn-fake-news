@echo off
title Truth GNN Analytics - Launcher
color 0B

echo ==================================================
echo       INICIANDO O TRUTH GNN ANALYTICS (TCC)
echo ==================================================
echo.

echo [1/2] Ligando o Motor Neural do Backend (FastAPI, PyTorch)...
start "Backend - FastAPI" cmd /k "cd frontend\api && ..\..\.venv\Scripts\python.exe -m uvicorn main:app --reload"

timeout /t 3 /nobreak >nul

echo [2/2] Construindo Dashboard Premium (Next.js, UI Apple-style)...
start "Frontend - Next.js" cmd /k "cd frontend\web && npm run dev"

echo.
echo ==================================================
echo   TUDO PRONTO! O sistema foi engatado com sucesso.
echo.
echo   - Sua API esta rodando silenciosamente na porta 8000.
echo   - Seu Dashboard visual pode ser acessado no navegador:
echo     http://localhost:3000
echo.
echo Pode fechar esta janela principal quando quiser. (Para desligar os servidores, feche as duas janelas pretas que abriram).
echo ==================================================
pause
