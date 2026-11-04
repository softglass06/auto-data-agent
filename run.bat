@echo off
chcp 65001 >nul
echo =====================================================================
echo   Starting Auto Data Agent (Streamlit Web Application)...
echo =====================================================================
echo.
streamlit run app.py
pause
