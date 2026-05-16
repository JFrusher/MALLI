# M.A.L.L.I. Dependency Management Guide

## Overview

This document provides industry-standard practices for managing dependencies across the M.A.L.L.I. dual-layer repository (ML + Mobile).

---

## 📋 Dependency Files Structure

```
MALLI/
├── .env.example              ← Environment template (NO SECRETS)
├── environment.yml           ← Conda environment spec
├── .gitignore               ← Git ignore rules
├── .gitattributes           ← Line ending management
├── ml/
│   └── requirements.txt     ← Python ML dependencies
└── pubspec.yaml             ← Flutter/Dart dependencies
```

---

## 🐍 Python / ML Dependencies

### Quick Start

```bash
# Using pip (recommended for development)
cd ml/
pip install -r requirements.txt

# Using conda (better for reproducibility)
conda env create -f ../environment.yml
conda activate malli
```

### File: `ml/requirements.txt`

**Purpose**: Specifies exact versions of all Python dependencies for reproducible builds.

**Structure**:
- Core ML libraries (TensorFlow, scikit-learn, OpenCV)
- Data processing (NumPy, Pillow, scikit-image)
- Visualization (Matplotlib, TensorBoard)
- Optional export formats (commented out, uncomment as needed)

**Usage**:
```bash
# Install exact versions
pip install -r ml/requirements.txt

# Update packages (interactive)
pip install --upgrade -r ml/requirements.txt

# Generate lock file for CI/CD
pip install pip-tools
pip-compile ml/requirements.txt
```

### File: `environment.yml`

**Purpose**: Conda environment specification for improved reproducibility and cross-platform compatibility.

**Why Conda over pip**:
- Pre-compiled binaries for scientific packages (faster)
- Better handling of system-level dependencies
- Easier multi-environment management
- Better support for CUDA/GPU

**Usage**:
```bash
# Create conda environment
conda env create -f environment.yml

# Activate
conda activate malli

# Update packages
conda update --all

# Export current environment
conda env export > environment.lock.yml
```

### Best Practices

✅ **Always use fixed versions** in requirements.txt for production
✅ **Pin major and minor versions** for development flexibility
✅ **Document why** complex dependencies are needed
✅ **Keep separate sections**: Core, Optional, Development
✅ **Update regularly** but test before deploying

### Optional Dependencies

**For ONNX export** (cross-platform model format):
```bash
pip install tf2onnx onnx onnxruntime
```

**For CoreML export** (iOS models, macOS only):
```bash
pip install coremltools
```

Uncomment in `requirements.txt` or `environment.yml` to use.

---

## 📦 Flutter / Mobile Dependencies

### File: `pubspec.yaml`

Located at repository root for Flutter package management.

**Key sections**:
- `name`: Package name (malli)
- `version`: Semantic versioning
- `environment`: SDK constraints (Flutter/Dart versions)
- `dependencies`: Runtime dependencies
- `dev_dependencies`: Development-only tools
- `flutter`: Asset and plugin configuration

**Current dependencies**:
- `camera`: Device camera access
- `path_provider`: File system paths
- `sqflite`: Local SQLite database
- `path`: Path manipulation utilities

### Dependency Management

```bash
# Get all dependencies
flutter pub get

# Update dependencies (respecting version constraints)
flutter pub upgrade

# Get specific dependency version
flutter pub add package_name:version

# Remove dependency
flutter pub remove package_name

# Generate lock file
flutter pub get  # pubspec.lock is auto-generated
```

### Adding New Dependencies

```bash
# Add direct dependency
flutter pub add camera_camera:^0.11.0

# Add dev dependency
flutter pub add --dev flutter_test

# Add with git repo
flutter pub add my_package --git https://github.com/user/my_package.git
```

### Upgrading Dependencies

```bash
# Check outdated packages
flutter pub outdated

# Upgrade all to latest allowed
flutter pub upgrade

# Upgrade specific package
flutter pub upgrade package_name

# Major version upgrade (breaking changes)
flutter pub upgrade package_name --major-versions
```

---

## 🔧 Environment Configuration

### File: `.env.example`

**Purpose**: Template for environment variables. Used for configuration that varies by developer/environment.

**Never commit**: `.env` (the actual file with secrets)
**Always commit**: `.env.example` (template with no secrets)

### Setup

```bash
# Copy example to local .env
cp .env.example .env

# Edit with your configuration
nano .env

# In Python, load environment variables
from dotenv import load_dotenv
import os

load_dotenv()  # Loads from .env
dataset_root = os.getenv('DATASET_ROOT')
```

### Categories of Variables

1. **Paths** - Where files are located
2. **Training** - Model hyperparameters
3. **Credentials** - API keys, tokens (NEVER in .env.example)
4. **Development** - Debug flags, log levels
5. **CI/CD** - Deployment configuration

---

## 🎯 Git Hygiene Configuration

### File: `.gitignore`

**Comprehensive coverage**:
- ✅ Python caches (`__pycache__/`, `.pytest_cache/`)
- ✅ Virtual environments (`venv/`, `env/`)
- ✅ Large model files (`*.h5`, `*.tflite`, `*.onnx`)
- ✅ Datasets (`datasets/`, `nih_data/`)
- ✅ Training outputs (`experiments/logs/`)
- ✅ IDE files (`.vscode/`, `.idea/`)
- ✅ Flutter/Android build (`build/`, `.gradle/`)
- ✅ iOS build (`Pods/`, `.xcworkspace/`)
- ✅ Environment files (`.env`)
- ✅ OS files (`.DS_Store`, `Thumbs.db`)

### File: `.gitattributes`

**Purpose**: Ensure consistent line endings across platforms (Windows/Mac/Linux)

**Sections**:
- Text files use Unix line endings (LF)
- Binary files remain untouched
- Git-specific handling for merging

---

## 🚀 Workflow Examples

### First-Time Setup

```bash
# Clone repository
git clone https://github.com/JFrusher/MALLI.git
cd MALLI

# Setup Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r ml/requirements.txt

# Setup Flutter
flutter pub get

# Configure environment
cp .env.example .env
# Edit .env with your paths and credentials

# Verify installation
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
flutter --version
```

### Development Workflow

```bash
# Make changes to code
# Edit ml/requirements.txt or pubspec.yaml as needed

# Test locally
python ml/train.py --help
flutter test

# Commit (only tracked files)
git add ml/src/ mobile/lib/
git commit -m "Add new feature"
git push
```

### Updating Dependencies

```bash
# Python
pip install --upgrade -r ml/requirements.txt
pip freeze > ml/requirements.txt

# Flutter
flutter pub upgrade
# pubspec.lock automatically updated

# Commit lock files
git add ml/requirements.txt pubspec.lock
git commit -m "Update dependencies"
```

### CI/CD Deployment

```bash
# Use lock files for reproducible builds
pip install -r ml/requirements.txt  # Uses exact versions from lock file
flutter pub get                      # Uses pubspec.lock

# Both ensure consistent environments across CI/CD
```

---

## 📊 Dependency Summary

### Python (ML Side)

**Core**: TensorFlow, NumPy, OpenCV, scikit-learn  
**Data**: Pillow, scikit-image, albumentations  
**Visualization**: Matplotlib, TensorBoard  
**Optional**: tf2onnx, coremltools (for export)

**Total**: ~15 main + 5 optional packages

### Flutter (Mobile Side)

**Camera**: Real-time device camera input  
**Storage**: Path provider, SQLite database  
**Utilities**: Path manipulation

**Total**: ~4 main packages

### System

**Python**: 3.10+  
**Flutter/Dart**: >=2.17.0  
**Node.js**: Not required (Flutter is self-contained)

---

## 🔒 Security Best Practices

### Secrets Management

✅ **DO**:
- Store secrets in `.env` (never committed)
- Use `.env.example` for templates
- Document what each variable is for
- Rotate API keys regularly
- Use environment variables in CI/CD

❌ **DON'T**:
- Commit `.env` files
- Hardcode credentials in code
- Share secrets in PRs or issues
- Use same passwords across environments
- Store secrets in version control

### Dependency Security

```bash
# Python: Check for vulnerabilities
pip install safety
safety check

# Flutter: Check for vulnerabilities
flutter pub outdated  # Shows security updates
flutter pub upgrade --major-versions
```

---

## 🛠️ Troubleshooting

### Python Import Errors

```bash
# Ensure venv is activated
source venv/bin/activate

# Reinstall requirements
pip install --force-reinstall -r ml/requirements.txt

# Check installed packages
pip list | grep tensorflow
```

### Flutter Dependency Issues

```bash
# Clear pub cache
flutter pub cache clean

# Reinstall
flutter pub get

# Check for conflicts
flutter pub deps --style=list
```

### Line Ending Issues (Windows/Mac/Linux)

```bash
# If files get messed up line endings
git config core.autocrlf input  # macOS/Linux
git config core.autocrlf true   # Windows

# Reapply attributes
git rm --cached -r .
git reset --hard
```

---

## 📚 Additional Resources

- **Python**: [pip documentation](https://pip.pypa.io/)
- **Conda**: [Conda documentation](https://docs.conda.io/)
- **Flutter**: [Pub.dev](https://pub.dev/)
- **Git**: [.gitignore patterns](https://git-scm.com/docs/gitignore)

---

## ✅ Checklist

Before committing to the repository:

- [ ] Virtual environment activated (Python side)
- [ ] All dependencies in requirements.txt (no pip install outside the file)
- [ ] No .env file committed (only .env.example)
- [ ] Large files in .gitignore (models, datasets)
- [ ] IDE files in .gitignore (.vscode, .idea)
- [ ] Lock files updated (pubspec.lock for Flutter)
- [ ] Environment variables documented in .env.example
- [ ] No credentials in any committed files
- [ ] Line endings consistent per .gitattributes

---

**Questions?** See documentation in the relevant `GUIDE.md` or `README.md` files.
