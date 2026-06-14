## Virtual Environment Setup

### Create Virtual Environment
```bash
python -m venv venv
```

### Activate Virtual Environment
```bash
.\venv\Scripts\Activate.ps1
```

### Remove Virtual Environment (if needed)
```bash
Remove-Item -Recurse -Force .\venv
```

## Required Libraries

```bash
pip install feature-engine
pip install ucimlrepo  # Required for sample data
```