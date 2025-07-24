#!/bin/bash

echo "============================================"
echo "Persistent Homology Classifier セットアップ"
echo "============================================"

# 仮想環境の確認
echo ""
echo "仮想環境の確認中..."
python3 -c "import sys; print('仮想環境:', sys.prefix != sys.base_prefix)"

if [[ -z "$VIRTUAL_ENV" ]]; then
    echo ""
    echo "⚠️  警告: 仮想環境が検出されませんでした"
    echo ""
    echo "仮想環境を作成して有効化してください:"
    echo "  conda create -n persistent_homology python=3.9 -y"
    echo "  conda activate persistent_homology"
    echo ""
    read -p "続行しますか？ (y/N): " continue
    if [[ "$continue" != "y" && "$continue" != "Y" ]]; then
        echo "セットアップを中止しました。"
        exit 1
    fi
fi

echo ""
echo "🔧 依存関係のインストール中..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ インストールに失敗しました。"
    echo "エラーを確認してください。"
    exit 1
fi

echo ""
echo "🧪 環境チェック実行中..."
python3 check_environment.py

echo ""
echo "✅ セットアップ完了！"
echo ""
echo "使用方法:"
echo "  python3 example_usage.py"
echo ""
