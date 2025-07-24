@echo off
echo ============================================
echo Persistent Homology Classifier セットアップ
echo ============================================

REM 仮想環境の確認
echo.
echo 仮想環境の確認中...
python -c "import sys; print('仮想環境:', sys.prefix != sys.base_prefix)"

if "%VIRTUAL_ENV%"=="" (
    echo.
    echo ⚠️  警告: 仮想環境が検出されませんでした
    echo.
    echo 仮想環境を作成して有効化してください:
    echo   conda create -n persistent_homology python=3.9 -y
    echo   conda activate persistent_homology
    echo.
    set /p continue="続行しますか？ (y/N): "
    if /i not "%continue%"=="y" (
        echo セットアップを中止しました。
        pause
        exit /b 1
    )
)

echo.
echo 🔧 依存関係のインストール中...
pip install -r requirements.txt

if %ERRORLEVEL% neq 0 (
    echo.
    echo ❌ インストールに失敗しました。
    echo エラーを確認してください。
    pause
    exit /b 1
)

echo.
echo 🧪 環境チェック実行中...
python check_environment.py

echo.
echo ✅ セットアップ完了！
echo.
echo 使用方法:
echo   python example_usage.py
echo.
pause
