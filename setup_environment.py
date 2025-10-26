import os
import sys
import subprocess
import importlib

def check_and_install_packages():
    """Check and install required packages"""
    required_packages = {
        'streamlit': 'streamlit',
        'pandas': 'pandas', 
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',
        'joblib': 'joblib',
        'langchain-groq': 'langchain_groq'
    }
    
    missing_packages = []
    
    for package, import_name in required_packages.items():
        try:
            importlib.import_module(import_name)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"Installing missing packages: {missing_packages}")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ Successfully installed {package}")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
    else:
        print("✅ All required packages are installed!")

def create_directories():
    """Create necessary directories"""
    directories = [
        "Uploaded_Datasets/Raw",
        "Uploaded_Datasets/Processed"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")



if __name__ == "__main__":
    print("🔧 Setting up TruLedger.AI environment...")
    check_and_install_packages()
    create_directories()
    print("🎉 Setup completed successfully!")