# Git Workflow & Repository Best Practices

## 📋 Repository Structure Overview

```
MALLI/
├── .gitignore               ← What NOT to commit (500+ rules)
├── .gitattributes           ← Line ending management (LF for text, binary for media)
├── .env.example             ← Environment template (NO SECRETS!)
├── environment.yml          ← Conda environment spec
├── DEPENDENCY_MANAGEMENT.md ← This file's companion
├── ml/
│   ├── requirements.txt     ← Python dependencies (pinned versions)
│   └── [training code]
├── mobile/
│   ├── lib/                 ← Flutter/Dart source
│   └── [platform specific]
└── pubspec.yaml             ← Flutter/Dart dependencies
```

---

## 🎯 Key Git Hygiene Rules

### ✅ ALWAYS COMMIT
- Source code (`.py`, `.dart`, `.java`, `.swift`)
- Configuration templates (`.env.example`, `config.yml`)
- Documentation (`README.md`, `*.md` files)
- Dependency manifests (`requirements.txt`, `pubspec.yaml`, `environment.yml`)
- GitHub workflows and CI/CD configs
- License and gitignore files

### ❌ NEVER COMMIT
- `.env` files (contain secrets/local config)
- Virtual environments (`venv/`, `env/`, `node_modules/`)
- Build artifacts (`build/`, `dist/`, `*.apk`, `*.ipa`)
- IDE files (`.vscode/settings.json`, `.idea/`)
- Large files (`*.h5`, `*.tflite`, `*.onnx`, datasets)
- Cache files (`__pycache__/`, `.pytest_cache/`, `.gradle/`)
- OS files (`.DS_Store`, `Thumbs.db`)
- Local logs and outputs

---

## 🚀 Daily Workflow

### Before Starting Work

```bash
# Activate Python environment
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows

# Get latest code
git pull origin main

# Install/update dependencies (only if they changed)
pip install -r ml/requirements.txt
flutter pub get
```

### During Development

```bash
# Make changes in your feature branch
git checkout -b feature/your-feature-name

# Stage only the files you changed (NOT .env, build/, etc.)
git status  # Review what will be committed

# Good commit staging
git add ml/train.py ml/src/export/pipeline.py
git add mobile/lib/screens/camera.dart
git add ml/requirements.txt  # If you added a dependency

# Commit with clear message
git commit -m "Add feature: description of what changed"

# Push to remote
git push origin feature/your-feature-name
```

### What NOT to Stage

```bash
# ❌ These will be rejected by .gitignore anyway:
git add venv/                    # Virtual environment
git add experiments/logs/         # Training outputs
git add .env                      # Local configuration
git add ml/**/__pycache__/        # Python cache

# Or if not in .gitignore, don't add them!
git status  # Use this to review before staging
```

---

## 🔍 Verification Commands

### Check Status
```bash
git status
# Shows what's staged, unstaged, and untracked
# Untracked files in red are GOOD (means they're ignored)
```

### Verify .gitignore Works
```bash
# Check if file is ignored
git check-ignore -v <filepath>

# Example: Should show these as IGNORED
git check-ignore -v venv/
git check-ignore -v experiments/logs/checkpoints/
git check-ignore -v .env
git check-ignore -v __pycache__/

# If not ignored, add to .gitignore!
```

### Review Before Committing
```bash
# See what changes you're about to commit
git diff --staged

# See all changes (staged + unstaged)
git diff
```

---

## 📦 Dependency Changes

### Adding a Python Package

```bash
# Install interactively
pip install new-package-name

# Add to requirements.txt (pinned version recommended)
# Example: Open ml/requirements.txt and add:
new-package-name==1.2.3

# Commit both code and requirements.txt
git add ml/your_code.py ml/requirements.txt
git commit -m "Add feature: implement X using new-package-name"

# Others will run: pip install -r ml/requirements.txt
```

### Adding a Flutter Package

```bash
# Add the package
flutter pub add camera:^0.10.0

# This updates pubspec.yaml and pubspec.lock automatically

# Commit both files
git add pubspec.yaml pubspec.lock
git commit -m "Add camera dependency"

# Others will run: flutter pub get
```

---

## 🛡️ Secrets Management

### Local Development Setup

```bash
# 1. Copy template (first time only)
cp .env.example .env

# 2. Edit with YOUR local configuration
# (Your IDE can open .env for editing)

# 3. NEVER COMMIT .env
# Git will refuse if you try to add it (it's in .gitignore)

# 4. Load in Python code
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('API_KEY')  # Reads from .env
```

### CI/CD Environment

In GitHub Actions (or your CI platform):
1. Go to Settings → Secrets
2. Add secret variables there
3. Access in workflows: `${{ secrets.API_KEY }}`
4. They're injected at runtime, never saved in the repo

---

## ⚠️ Common Mistakes & How to Fix

### Mistake 1: Committed `.env` with secrets

```bash
# ❌ WRONG - If you did this:
cp .env.example .env
# Edited with API keys
git add .env
git commit -m "Add config"

# ✅ FIX:
# 1. Remove from Git history
git rm --cached .env
git commit -m "Remove .env (had secrets)"

# 2. Make sure .gitignore has: .env
# 3. All past commits with secrets? Contact admin to rotate API keys

# 4. Add .env to .gitignore if not there
echo ".env" >> .gitignore
git add .gitignore
git commit -m "Ignore .env files"
```

### Mistake 2: Committed large model files

```bash
# ❌ If you accidentally committed a large file:
git rm --cached models/weights.h5
git commit -m "Remove large model file (should be in .gitignore)"

# ✅ Then add to .gitignore:
echo "*.h5" >> .gitignore
echo "*.tflite" >> .gitignore
echo "*.onnx" >> .gitignore
git add .gitignore
git commit -m "Add model files to .gitignore"
```

### Mistake 3: Forgot to activate venv

```bash
# ❌ Wrong - dependencies install globally:
pip install tensorflow  # Without venv activated

# ✅ Right - Always activate first:
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows
pip install tensorflow    # Now installs in venv only
```

### Mistake 4: Committing IDE settings

```bash
# ❌ Accidentally added .vscode/:
git add .vscode/
git commit -m "Add VSCode settings"

# ✅ FIX - Remove IDE files:
git rm --cached .vscode/settings.json
git commit -m "Remove IDE-specific files"

# Already in .gitignore but was tracked? This forces removal:
git rm -r --cached .vscode/
git commit -m "Untrack .vscode directory"
```

---

## 📊 Example Commits

### Good Commit
```
git add ml/train.py ml/src/export/pipeline.py
git add ml/requirements.txt
git commit -m "Implement multi-format model export (TFLite, ONNX, CoreML)"

# Message explains WHAT changed and WHY
# Only source code and dependency changes included
# No large files, no secrets, no build artifacts
```

### Bad Commits to Avoid
```
# ❌ Vague message
git commit -m "updates"

# ❌ Large file included
git add models/weights.h5
git commit -m "update model"

# ❌ Secret committed
git add .env
git commit -m "add config"

# ❌ IDE settings
git add .vscode/settings.json
git commit -m "VSCode config"

# ❌ Multiple unrelated changes
git add ml/train.py mobile/lib/ .env venv/
git commit -m "various updates"
```

---

## 🔐 Pre-Commit Checklist

Before running `git commit`:

- [ ] **No secrets** - Doesn't contain API keys, passwords, credentials
- [ ] **No .env** - `.env` file is NOT staged
- [ ] **No large files** - No model weights, datasets, archives
- [ ] **No IDE files** - `.vscode/`, `.idea/` not included
- [ ] **No cache** - `__pycache__/`, `.pytest_cache/` not included
- [ ] **No build artifacts** - `build/`, `dist/`, Gradle cache not included
- [ ] **Dependencies pinned** - If changed `requirements.txt`, versions are locked
- [ ] **Message clear** - Commit message explains the change
- [ ] **Only YOUR changes** - Not reverting others' work
- [ ] **Code tested** - Doesn't break existing functionality

---

## 🚨 Emergency: Oops! I Committed Something Bad!

### If not yet pushed:

```bash
# Uncommit without losing changes:
git reset --soft HEAD~1
git status  # Files will be unstaged but kept

# Then re-stage only the good files:
git add good_file.py
git commit -m "Fixed commit (removed bad files)"
```

### If already pushed:

```bash
# Create a new commit that removes the bad file:
git rm --cached bad_file  # Remove from version control
git commit -m "Remove bad file"
git push origin main

# For secrets that were exposed:
# 1. Rotate all credentials immediately
# 2. Clear Git history (if critical):
#    - Contact repo admin
#    - May need to rewrite history with git filter-branch
# 3. Never reuse the same credentials
```

---

## 📚 Setup Guide

### First Time Only

```bash
# Clone the repo
git clone https://github.com/JFrusher/MALLI.git
cd MALLI

# Setup Python
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate on Windows
pip install -r ml/requirements.txt

# Setup Flutter
flutter pub get

# Setup Git
cp .env.example .env
# Edit .env with your configuration

# Verify setup
python -c "import tensorflow; print('Python OK')"
flutter --version
git status  # Should show no modifications
```

### Ongoing Workflow

```bash
# Every day:
git pull origin main
pip install -r ml/requirements.txt  # If dependencies changed
flutter pub get                      # If pubspec changed

# Make your changes in a feature branch
git checkout -b feature/name

# Commit regularly with clear messages
git add ml/your_file.py
git commit -m "Add feature: description"

# Push and create PR
git push origin feature/name
```

---

## 🎓 Key Takeaways

1. **`.gitignore` is your friend** - It protects you from committing secrets and large files
2. **Commit messages matter** - Future you will thank present you for clear descriptions
3. **Virtual environments are local** - Don't commit them, document in `requirements.txt`
4. **Secrets go in `.env`** - Never in code, never in version control
5. **Use `git status`** - Before every commit, verify what you're committing
6. **Lock files matter** - `requirements.txt` and `pubspec.lock` ensure reproducible builds
7. **One feature per commit** - Easier to review, easier to revert if needed

---

For more details, see: [DEPENDENCY_MANAGEMENT.md](DEPENDENCY_MANAGEMENT.md)
