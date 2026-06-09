# 📚 Git Hygiene Configuration - Master Reference

## Overview

Your M.A.L.L.I. repository now has **comprehensive, production-grade Git hygiene** covering both ML and Mobile development workflows.

---

## 🗂️ Configuration Files Reference

### Core Git Configuration Files

#### 1. `.gitignore` (400+ rules)
**Location**: Repository root  
**Status**: ✅ Configured  
**Purpose**: Prevents commits of secrets, large files, and development clutter

**What It Protects**:
- **Secrets**: `.env`, credentials, API keys
- **Large Files**: Model weights (`*.h5`, `*.tflite`, `*.onnx`), datasets
- **Python Clutter**: `__pycache__/`, `venv/`, `.pytest_cache/`
- **Build Artifacts**: `build/`, `dist/`, `*.apk`, `*.ipa`
- **IDE Files**: `.vscode/`, `.idea/`, `*.sublime-*`
- **OS Files**: `.DS_Store`, `Thumbs.db`
- **Flutter/Mobile**: `.gradle/`, `Pods/`, `.xcworkspace/`

**How to Use**:
```bash
# Check if a file is ignored:
git check-ignore -v filename

# Example: Should return a match
git check-ignore -v venv/

# Example: Should return nothing (not ignored)
git check-ignore -v ml/train.py
```

---

#### 2. `.gitattributes` (Line ending management)
**Location**: Repository root  
**Status**: ✅ Configured  
**Purpose**: Ensures consistent line endings across Windows/Mac/Linux

**What It Does**:
- Text files always use Unix line endings (LF)
- Binary files remain untouched
- Prevents `^M` characters and merge conflicts
- Handles 30+ file types automatically

**Coverage**:
- ✅ Source code: `.py`, `.dart`, `.java`, `.swift`, `.js`, `.ts`
- ✅ Config: `.yml`, `.yaml`, `.json`, `.toml`, `.ini`
- ✅ Scripts: `.sh`, `Makefile`, `.bash`
- ✅ Docs: `.md`, `.rst`, `.txt`
- ✅ Binary: Models, images, archives (untouched)

**How to Use**: It's automatic! Git handles it when you push/pull.

---

#### 3. `.env.example` (Configuration template)
**Location**: Repository root  
**Status**: ✅ Configured  
**Purpose**: Documents configuration structure (safe to commit, no secrets)

**What It Contains**:
- ML paths (dataset root, logs, models)
- Training parameters (batch size, learning rate, epochs)
- Mobile config (Flutter app paths, Android/iOS settings)
- Development flags (verbosity, debug mode)
- **IMPORTANT**: No actual secrets or credentials

**How to Use**:
```bash
# Copy to local configuration (first time setup)
cp .env.example .env

# Edit with YOUR local settings
nano .env  # or use your editor

# In Python, load it:
from dotenv import load_dotenv
load_dotenv()  # Reads from .env file

# .env is in .gitignore, so it's never committed
```

---

### Dependency Management Files

#### 4. `ml/requirements.txt` (Python dependencies)
**Location**: `ml/` directory  
**Status**: ✅ Configured  
**Purpose**: Specifies exact versions of all Python packages

**Current Dependencies**:
```
tensorflow==2.13.1          # Core ML framework
numpy==1.24.3               # Numerical computing
opencv-python==4.8.0.74    # Computer vision
torch==2.1.0                # Alternative ML framework
ultralytics==8.0.181        # YOLOv8 object detection
scikit-learn==1.3.0         # ML utilities
... and more
```

**Optional Dependencies** (uncomment to use):
```
# tf2onnx==1.14.0            # ONNX export support
# coremltools==7.0           # CoreML export support (macOS)
```

**How to Use**:
```bash
# Install exact versions
pip install -r ml/requirements.txt

# Add new package
pip install new-package
pip freeze | grep new-package  # Get version
# Edit requirements.txt: add "new-package==1.2.3"

# Update all packages
pip install --upgrade -r ml/requirements.txt

# Generate lock file for CI/CD
pip-compile ml/requirements.txt  # Creates requirements.lock
```

---

#### 5. `environment.yml` (Conda environment)
**Location**: Repository root  
**Status**: ✅ Configured  
**Purpose**: Alternative to pip - better for reproducibility and scientific computing

**Why Use Conda**:
- ✅ Pre-compiled binaries (faster)
- ✅ Better CUDA/GPU support
- ✅ Handles system dependencies
- ✅ Cross-platform reproducibility

**How to Use**:
```bash
# Create conda environment
conda env create -f environment.yml

# Activate
conda activate malli

# Update packages
conda update --all

# Export your current environment
conda env export > environment.lock.yml
```

**Choice**: Use either pip OR conda, not both:
- **pip** (simpler, standard): Good for most projects
- **conda** (scientific computing): Better for ML with CUDA

---

#### 6. `pubspec.yaml` (Flutter/Dart dependencies)
**Location**: Repository root  
**Status**: ✅ Configured  
**Purpose**: Specifies Flutter and Dart package versions

**Current Dependencies**:
```yaml
camera: ^0.10.0+1        # Device camera access
path_provider: ^2.0.15   # File system paths
sqflite: ^2.2.8          # Local SQLite database
```

**How to Use**:
```bash
# Get all dependencies
flutter pub get

# Add new package
flutter pub add camera:^0.11.0

# Update packages
flutter pub upgrade

# Check for outdated packages
flutter pub outdated
```

---

### Documentation Files

#### 7. `DEPENDENCY_MANAGEMENT.md` (900+ lines)
**Purpose**: Comprehensive guide for managing dependencies on both sides

**Sections**:
- Python/pip workflow
- Conda for scientific computing
- Flutter/Dart package management
- Environment configuration (`.env`)
- Adding/updating dependencies
- Secrets management
- Security best practices
- Troubleshooting

**When to Read**: When you need to add/update packages or understand dependency management

---

#### 8. `GIT_WORKFLOW.md` (600+ lines)
**Purpose**: Daily development practices and Git workflow

**Sections**:
- Repository structure overview
- Commit/ignore rules
- Daily workflow examples
- Verification commands
- Common mistakes and fixes
- Pre-commit checklist
- Emergency procedures

**When to Read**: When you're about to commit or have Git questions

---

#### 9. `GIT_HYGIENE_SETUP.md` (400+ lines)
**Purpose**: Implementation summary and reference

**Sections**:
- What's been implemented
- Before/after comparison
- Quick start guide
- Protection features explained
- Usage examples
- Emergency procedures
- CI/CD integration

**When to Read**: To understand the full setup and benefits

---

#### 10. `QUICK_START_GIT_HYGIENE.md` (This checklist!)
**Purpose**: Quick reference and checklist

**Sections**:
- Implementation status
- First-time setup
- Daily workflow
- Common issues
- Verification commands
- Documentation map

**When to Read**: When you need a quick reference or checklist

---

## 🔄 Workflow Decision Tree

```
I want to...              Then...
────────────────────────────────────────────────────────
Add a Python package      → See: DEPENDENCY_MANAGEMENT.md → Python section
Add a Flutter package     → See: DEPENDENCY_MANAGEMENT.md → Flutter section
Make a commit             → See: GIT_WORKFLOW.md → Daily workflow
Understand .gitignore     → See: .gitignore (in repo) + GIT_WORKFLOW.md
Set up for first time     → See: QUICK_START_GIT_HYGIENE.md → First Time Setup
Fix a Git mistake         → See: GIT_WORKFLOW.md → Common Mistakes & Fixes
Check if file is ignored  → Run: git check-ignore -v filename
Set up local config       → cp .env.example .env, then edit
Use conda instead of pip  → See: DEPENDENCY_MANAGEMENT.md → Conda section
Understand line endings   → See: .gitattributes (in repo)
```

---

## 🚀 Essential Commands

### Verification
```bash
git check-ignore -v <file>        # Is this file ignored?
git status                         # What will be committed?
git diff --staged                  # Review changes before commit
```

### Setup
```bash
cp .env.example .env              # Create local config
pip install -r ml/requirements.txt # Install Python deps
flutter pub get                    # Install Flutter deps
```

### Commits
```bash
git add <file>                    # Stage specific file
git commit -m "description"       # Commit with message
git push origin branch            # Push to remote
```

### Troubleshooting
```bash
git reset --soft HEAD~1           # Undo last commit (keep changes)
git rm --cached <file>            # Remove from git (keep local)
git show <commit>:<file>          # View file at specific commit
```

---

## 📋 Checklist: Before Every Commit

- [ ] Activated venv: `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
- [ ] Reviewed changes: `git status` shows only YOUR changes
- [ ] No secrets: `.env` NOT in staged files
- [ ] No large files: Models, datasets, archives not staged
- [ ] No IDE files: `.vscode/`, `.idea/` not staged
- [ ] No cache: `__pycache__/`, `.pytest_cache/` not staged
- [ ] Dependencies updated: If you changed requirements, updated the file
- [ ] Clear message: Commit message explains WHAT and WHY
- [ ] Code tested: Doesn't break existing functionality

---

## 🎯 File Organization

### For Python/ML Development
```
ml/
├── requirements.txt       ← Add/update here for new packages
├── train.py              ← Your code (commit this)
├── src/                  ← Your modules (commit these)
└── experiments/logs/     ← Training outputs (IGNORED, don't commit)
```

### For Flutter/Mobile Development
```
mobile/
├── lib/                  ← Your Dart code (commit this)
├── pubspec.yaml          ← Update when adding packages
└── build/                ← Build artifacts (IGNORED, don't commit)
```

### For Configuration
```
MALLI/
├── .env.example          ← Template (COMMIT THIS)
├── .env                  ← Your local config (IGNORED, don't commit)
├── environment.yml       ← Python environment (COMMIT THIS)
└── pubspec.yaml          ← Flutter packages (COMMIT THIS)
```

---

## 🔐 Secrets Handling

### ❌ WRONG
```python
# In your code:
api_key = "sk-abc123def456"  # WRONG: hardcoded!
```

### ✅ RIGHT
```bash
# In .env:
API_KEY=sk-abc123def456

# In your code:
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv('API_KEY')
```

### ✅ FOR CI/CD
```yaml
# In GitHub Actions (Settings → Secrets):
# Add: DEPLOY_KEY, API_KEY, etc.

# In workflow:
env:
  API_KEY: ${{ secrets.API_KEY }}  # Injected at runtime
```

---

## 🎓 Learning Path

### Day 1: Setup
1. Read: [QUICK_START_GIT_HYGIENE.md](QUICK_START_GIT_HYGIENE.md)
2. Run: First-time setup commands
3. Test: Verify everything works

### Week 1: Daily Workflow
1. Read: [GIT_WORKFLOW.md](GIT_WORKFLOW.md) → Daily Workflow section
2. Practice: Make commits, verify with `git status`
3. Refer: To checklist before each commit

### As Needed
1. **Adding packages**: [DEPENDENCY_MANAGEMENT.md](DEPENDENCY_MANAGEMENT.md)
2. **Git problems**: [GIT_WORKFLOW.md](GIT_WORKFLOW.md) → Common Mistakes
3. **Setup help**: [QUICK_START_GIT_HYGIENE.md](QUICK_START_GIT_HYGIENE.md) → First Time Setup

---

## ✨ Key Benefits

### Security
✅ Secrets can't be accidentally committed (`.env` in `.gitignore`)
✅ No API keys in version history
✅ CI/CD injects secrets at runtime

### Clean Repository
✅ No large files (models, datasets ignored)
✅ No build artifacts (caches, compiled files ignored)
✅ No development clutter (IDE files ignored)

### Reproducibility
✅ Pinned dependency versions
✅ Lock files for exact reproduction
✅ Consistent line endings across platforms

### Developer Experience
✅ Clear setup instructions
✅ Git protects against common mistakes
✅ Documentation for all workflows

---

## 🆘 Quick Emergency Guide

| Problem | Solution | Details |
|---------|----------|---------|
| Committed `.env` with secrets | `git reset --soft HEAD~1` then recommit without it | See: GIT_WORKFLOW.md |
| Committed large file | `git rm --cached file` then recommit | See: GIT_WORKFLOW.md |
| Line ending issues | Already fixed by `.gitattributes` | No action needed |
| Can't remember what commits | `git log --oneline` | Shows recent commits |
| Need to undo commit (not pushed) | `git reset --soft HEAD~1` | Keeps your changes |
| Need to see what was ignored | `git check-ignore -v <file>` | Shows ignore rule |

---

## 📞 Documentation Map

```
Question?                           Reference
──────────────────────────────────────────────────────────
How do I add a new package?         DEPENDENCY_MANAGEMENT.md
What should I commit?                GIT_WORKFLOW.md
How do I set up locally?            QUICK_START_GIT_HYGIENE.md
Why is file X ignored?               git check-ignore -v <file>
How do I configure secrets?          .env.example (copy & edit)
What's in .gitignore?               .gitignore (in repo)
I made a Git mistake!               GIT_WORKFLOW.md → Emergencies
Can I use Conda instead of pip?     DEPENDENCY_MANAGEMENT.md → Conda
How do line endings work?           .gitattributes (in repo)
New developer setup?                QUICK_START_GIT_HYGIENE.md
```

---

## 🎯 Final Checklist: You're Ready!

- [ ] ✅ `.gitignore` configured with 400+ rules
- [ ] ✅ `.gitattributes` handles line endings automatically
- [ ] ✅ `.env.example` documents configuration structure
- [ ] ✅ `requirements.txt` pins Python versions
- [ ] ✅ `environment.yml` provides Conda alternative
- [ ] ✅ `pubspec.yaml` manages Flutter dependencies
- [ ] ✅ Documentation guides created (900+ lines total)
- [ ] ✅ Security best practices documented
- [ ] ✅ Emergency procedures documented
- [ ] ✅ Workflow decisions mapped

**Your repository is now production-ready! 🚀**

---

**Questions?** Check the documentation map above or read the appropriate guide.
