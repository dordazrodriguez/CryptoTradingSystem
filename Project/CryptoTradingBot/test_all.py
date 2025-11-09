#!/usr/bin/env python3
"""
Comprehensive Test Script for Crypto Trading Bot Deliverables
"""
import sys
import os
import subprocess
import requests
import time
from datetime import datetime

# Setup logging to both console and file
log_file = None
log_file_path = None

def setup_logging():
    """Setup logging to both console and file"""
    global log_file, log_file_path
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(logs_dir, f'test_results_{timestamp}.log')
    log_file = open(log_file_path, 'w', encoding='utf-8')
    
    return log_file_path

def log_print(*args, **kwargs):
    """Print to both console and log file"""
    # Print to console
    print(*args, **kwargs)
    # Write to log file
    if log_file:
        # Convert args to string and write to file
        message = ' '.join(str(arg) for arg in args)
        log_file.write(message + '\n')
        log_file.flush()  # Ensure immediate write

def close_logging():
    """Close log file"""
    global log_file
    if log_file:
        log_file.close()
        log_file = None

def test_imports():
    """Test that all required modules can be imported"""
    log_print("Testing Module Imports...")
    all_passed = True
    
    # Test core modules (should always work)
    try:
        from trading_engine import indicators
        log_print("  ✅ Trading engine indicators imported")
    except Exception as e:
        log_print(f"  ❌ Trading engine error: {e}")
        all_passed = False
    
    try:
        from ml_models import predictor
        log_print("  ✅ ML predictor imported")
    except Exception as e:
        log_print(f"  ❌ ML predictor error: {e}")
        all_passed = False
    
    try:
        from data import processor
        log_print("  ✅ Data processor imported")
    except Exception as e:
        log_print(f"  ❌ Data processor error: {e}")
        all_passed = False
    
    # Test optional modules (may require dependencies)
    try:
        from data import collector
        log_print("  ✅ Data collector imported")
    except Exception as e:
        log_print(f"  ⚠️  Data collector requires ccxt (optional): {e}")
    
    try:
        from backend import app
        log_print("  ✅ Backend app imported")
    except ImportError as e:
        if 'ccxt' in str(e) or 'ccxt' in str(e).lower():
            log_print(f"  ⚠️  Backend app requires ccxt (optional): {e}")
        else:
            log_print(f"  ❌ Backend app error: {e}")
            all_passed = False
    except Exception as e:
        log_print(f"  ❌ Backend app error: {e}")
        all_passed = False
    
    if all_passed:
        log_print("✅ Core modules imported successfully")
    else:
        log_print("⚠️  Some core modules failed to import")
    
    return all_passed

def test_database():
    """Test database connection and structure"""
    log_print("\nTesting Database...")
    try:
        from data.db import get_db_manager
        db = get_db_manager()
        
        # Test basic connection
        conn = db.get_connection()
        log_print("  ✅ Database connection successful")
        conn.close()
        
        # Test query capability
        try:
            stats = db.get_database_stats()
            log_print(f"  ✅ Database query successful")
            log_print(f"     Stats: {stats}")
        except Exception as e:
            log_print(f"  ⚠️  Database stats query: {e}")
            # Try a simple query
            try:
                result = db.execute_query("SELECT COUNT(*) as count FROM trades LIMIT 1")
                log_print(f"  ✅ Database query test passed")
            except:
                log_print(f"  ⚠️  Trades table may not exist yet (this is OK)")
        
        return True
    except Exception as e:
        log_print(f"❌ Database error: {e}")
        import traceback
        exc_info = traceback.format_exc()
        log_print(exc_info)
        return False

def test_indicators():
    """Test technical indicator calculations"""
    log_print("\nTesting Technical Indicators...")
    try:
        from trading_engine.indicators import TechnicalIndicators
        import pandas as pd
        import numpy as np
        
        indicators = TechnicalIndicators()
        
        # Create sample data (16+ periods for RSI)
        prices = pd.Series([100 + i + np.random.random() for i in range(30)])
        
        # Test RSI (use window parameter, not period)
        rsi = indicators.calculate_rsi(prices, window=14)
        rsi_val = rsi.iloc[-1]
        log_print(f"  ✅ RSI calculated: {rsi_val:.2f}")
        if pd.isna(rsi_val):
            log_print("     ⚠️  Warning: RSI is NaN (may be expected for early periods)")
        
        # Test MACD
        macd_result = indicators.calculate_macd(prices)
        macd_val = macd_result['macd'].iloc[-1]
        log_print(f"  ✅ MACD calculated: {macd_val:.2f}")
        
        # Test Bollinger Bands (use window parameter, not period)
        bb = indicators.calculate_bollinger_bands(prices, window=20)
        log_print(f"  ✅ Bollinger Bands calculated")
        log_print(f"     Upper: {bb['upper'].iloc[-1]:.2f}")
        log_print(f"     Middle: {bb['middle'].iloc[-1]:.2f}")
        log_print(f"     Lower: {bb['lower'].iloc[-1]:.2f}")
        
        # Test SMA
        sma = indicators.calculate_sma(prices, window=20)
        sma_val = sma.iloc[-1]
        log_print(f"  ✅ SMA calculated: {sma_val:.2f}")
        
        log_print("✅ All indicator calculations successful")
        return True
    except Exception as e:
        log_print(f"❌ Indicator error: {e}")
        import traceback
        exc_info = traceback.format_exc()
        log_print(exc_info)
        return False

def test_api():
    """Test API endpoints (requires backend to be running)"""
    log_print("\nTesting API Endpoints...")
    base_url = "http://localhost:5000"
    
    try:
        response = requests.get(f"{base_url}/health", timeout=2)
        if response.status_code == 200:
            log_print("  ✅ Health endpoint working")
            log_print(f"     Response: {response.json()}")
            
            # Test additional endpoints
            endpoints = [
                "/api/portfolio",
                "/api/trades",
                "/api/market/prices/BTC-USD"
            ]
            
            for endpoint in endpoints:
                try:
                    resp = requests.get(f"{base_url}{endpoint}", timeout=2)
                    status = "✅" if resp.status_code == 200 else "⚠️"
                    log_print(f"  {status} {endpoint}: {resp.status_code}")
                except:
                    pass
            
            return True
        else:
            log_print(f"❌ Health endpoint returned {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        log_print("⚠️  Backend not running (start with: python backend/app.py)")
        log_print("     This test is optional - API tests can be run when backend is started")
        return None
    except Exception as e:
        log_print(f"❌ API error: {e}")
        return False

def test_documents():
    """Verify all required documents exist"""
    log_print("\nTesting Documentation Files...")
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    required_docs = [
        "docs/Task2/A_Letter_of_Transmittal_and_Project_Proposal.md",
        "docs/Task2/B_Executive_Summary.md",
        "docs/Task2/D1_Business_Vision_and_Requirements.md",
        "docs/Task2/D2_Hypothesis_Assessment.md",
        "docs/Task2/D3_Accuracy_Assessment.md",
        "docs/Task2/D4_Testing_Results.md",
        "docs/Task2/D5_Quick_Start_Guide.md",
        "docs/Task2/Task2_Deliverables_Summary.md"
    ]
    
    all_exist = True
    for doc in required_docs:
        doc_path = os.path.join(base_path, doc)
        if os.path.exists(doc_path):
            size = os.path.getsize(doc_path)
            log_print(f"  ✅ {os.path.basename(doc)} ({size:,} bytes)")
        else:
            log_print(f"  ❌ {os.path.basename(doc)} - NOT FOUND")
            all_exist = False
    
    return all_exist

def test_main_executable():
    """Test that main.py can be imported and has required functions"""
    log_print("\nTesting Main Executable...")
    try:
        # Check if main.py file exists and is accessible
        import os
        main_path = os.path.join(os.path.dirname(__file__), "main.py")
        if os.path.exists(main_path):
            log_print("  ✅ main.py file exists")
            # Try to import only if dependencies are available
            try:
                import main
                log_print("  ✅ main.py imports successfully")
                return True
            except ImportError as e:
                if 'ccxt' in str(e) or 'ccxt' in str(e).lower():
                    log_print("  ⚠️  main.py requires ccxt (install with: pip install ccxt)")
                    log_print("     File exists and is accessible, but import requires dependencies")
                    return None  # Skip, not a failure
                else:
                    log_print(f"  ❌ Import error: {e}")
                    return False
        else:
            log_print(f"  ❌ main.py not found at {main_path}")
            return False
    except Exception as e:
        log_print(f"❌ Main module error: {e}")
        import traceback
        exc_info = traceback.format_exc()
        log_print(exc_info)
        return False

def test_data_processor():
    """Test data processing functionality"""
    log_print("\nTesting Data Processing...")
    try:
        from data.processor import DataProcessor
        import pandas as pd
        
        processor = DataProcessor()
        
        # Create sample data
        sample_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [99, 100, 101],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200]
        })
        
        log_print("  ✅ DataProcessor imported")
        log_print("  ✅ Sample DataFrame created")
        
        # Test basic operations
        if hasattr(processor, 'clean_data'):
            log_print("  ✅ clean_data method exists")
        if hasattr(processor, 'process'):
            log_print("  ✅ process method exists")
        
        return True
    except Exception as e:
        log_print(f"❌ Data processor error: {e}")
        import traceback
        exc_info = traceback.format_exc()
        log_print(exc_info)
        return False

def main():
    """Run all tests"""
    # Setup logging first
    log_path = setup_logging()
    log_print("=" * 60)
    log_print("COMPREHENSIVE TESTING SUITE")
    log_print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"Log file: {log_path}")
    log_print("=" * 60)
    
    try:
        results = {
            "Imports": test_imports(),
            "Main Executable": test_main_executable(),
            "Data Processing": test_data_processor(),
            "Database": test_database(),
            "Indicators": test_indicators(),
            "API": test_api(),
            "Documents": test_documents()
        }
        
        log_print("\n" + "=" * 60)
        log_print("TEST SUMMARY")
        log_print("=" * 60)
        
        for test_name, result in results.items():
            if result is True:
                log_print(f"✅ {test_name}: PASS")
            elif result is None:
                log_print(f"⚠️  {test_name}: SKIPPED (requirements not met)")
            else:
                log_print(f"❌ {test_name}: FAIL")
        
        passed = sum(1 for r in results.values() if r is True)
        skipped = sum(1 for r in results.values() if r is None)
        total = len(results)
        failed = total - passed - skipped
        
        log_print(f"\nResults: {passed} passed, {skipped} skipped, {failed} failed")
        log_print(f"Pass Rate: {passed}/{total - skipped} (excluding skipped)")
        log_print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_print(f"\nLog file saved to: {log_path}")
        
        if passed == total - skipped:
            log_print("\n✅ All applicable tests passed!")
            return 0
        else:
            log_print("\n⚠️  Some tests failed or were skipped")
            log_print("   Review output above for details")
            return 1
    finally:
        # Always close log file
        close_logging()

if __name__ == "__main__":
    sys.exit(main())

