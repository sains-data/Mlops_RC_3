# tests/test_app.py

def test_basic_math():
    """Tes dummy untuk memastikan pytest berjalan."""
    assert 1 + 1 == 2

def test_config_exists():
    """Memastikan file config ada."""
    from pathlib import Path
    import os
    
    # Cek apakah file config data.yaml ada
    # Kita asumsikan tes dijalankan dari root directory
    config_path = Path("configs/data.yaml")
    
    # Assert True jika file ada
    assert config_path.exists() == True