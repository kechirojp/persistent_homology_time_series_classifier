#!/usr/bin/env python3
"""
Environment Check for Persistent Homology Classifier

持続ホモロジー分類器の環境チェックスクリプト
"""

import sys
import importlib.util

def check_python_version():
    """Python バージョンの確認"""
    print("🐍 Python バージョン確認")
    version = sys.version_info
    print(f"   バージョン: {version.major}.{version.minor}.{version.micro}")
    
    if version >= (3, 8):
        print("   ✅ Python 3.8以上です")
        return True
    else:
        print("   ❌ Python 3.8以上が必要です")
        return False

def check_core_dependencies():
    """コア依存関係の確認"""
    print("\n📦 コア依存関係の確認")
    
    # 主要パッケージのチェック
    packages = [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("sklearn", "scikit-learn"),
        ("gudhi", "gudhi"),
        ("networkx", "networkx"),
        ("matplotlib", "matplotlib"),
        ("pandas", "pandas"),
        ("tqdm", "tqdm")
    ]
    
    all_good = True
    
    for display_name, import_name in packages:
        try:
            spec = importlib.util.find_spec(import_name)
            if spec is not None:
                module = importlib.import_module(import_name)
                version = getattr(module, '__version__', 'unknown')
                print(f"   ✅ {display_name}: v{version}")
            else:
                print(f"   ❌ {display_name}: インストールされていません")
                all_good = False
                
        except ImportError:
            print(f"   ❌ {display_name}: インポートエラー")
            all_good = False
        except Exception as e:
            print(f"   ⚠️ {display_name}: チェックエラー ({e})")
    
    return all_good

def check_optional_dependencies():
    """オプション依存関係の確認"""
    print("\n📦 オプション依存関係の確認")
    
    optional_packages = [
        ("seaborn", "seaborn", "可視化の拡張"),
        ("jupyterlab", "jupyterlab", "Jupyter Notebook環境")
    ]
    
    for display_name, import_name, description in optional_packages:
        try:
            spec = importlib.util.find_spec(import_name)
            if spec is not None:
                module = importlib.import_module(import_name)
                version = getattr(module, '__version__', 'unknown')
                print(f"   ✅ {display_name}: v{version} ({description})")
            else:
                print(f"   ⚪ {display_name}: インストールされていません")
        except Exception:
            print(f"   ⚪ {display_name}: チェックできません")

def test_basic_functionality():
    """基本機能のテスト"""
    print("\n🧪 基本機能テスト")
    
    try:
        import numpy as np
        import networkx as nx
        
        # NumPy配列作成テスト
        x = np.random.randn(10, 3)
        print("   ✅ NumPy配列作成 OK")
        
        # NetworkXグラフ作成テスト
        G = nx.Graph()
        G.add_edge(1, 2)
        print("   ✅ NetworkXグラフ作成 OK")
        
        # GUDHIテスト（可能な場合）
        try:
            import gudhi as gd
            simplex_tree = gd.SimplexTree()
            simplex_tree.insert([0, 1])
            print("   ✅ GUDHI基本機能 OK")
        except ImportError:
            print("   ⚠️ GUDHI未インストール")
            
        return True
        
    except Exception as e:
        print(f"   ❌ テスト失敗: {e}")
        return False

def main():
    """メイン実行関数"""
    print("=" * 60)
    print("🔍 Persistent Homology Classifier 環境チェック")
    print("=" * 60)
    
    checks = []
    
    # 各チェックを実行
    checks.append(check_python_version())
    checks.append(check_core_dependencies())
    check_optional_dependencies()
    checks.append(test_basic_functionality())
    
    # 総合結果
    print("\n" + "=" * 60)
    print("📋 総合結果")
    print("=" * 60)
    
    if all(checks):
        print("🎉 すべてのチェックが完了しました！")
        print("   このシステムでPersistent Homology Classifierを実行できます。")
    else:
        print("⚠️ いくつかの問題が見つかりました。")
        print("   必要なパッケージをインストールしてください:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()
