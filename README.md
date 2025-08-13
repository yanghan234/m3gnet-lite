# Python Project Template

This is a template for quickly creating Python projects with modern tooling and best practices.

## ⚡ Quick Reference - What You Need to Change

**After creating your project from this template, update these 3 things:**

1. **`pyproject.toml`** - Replace `python_project_template` placeholders with your project name
2. **`src/python_project_template/`** - Rename this directory to match your package name  
3. **`README.md`** - Update the title and description

**Example**: If your project is called "my-awesome-project":
- `pyproject.toml`: `name = "my-awesome-project"`
- Directory: `src/python_project_template/` → `src/my_awesome_project/`
- `README.md`: `# my-awesome-project`

**Note**: The template uses vanilla names that will be automatically updated by the setup script.

**🚀 Quick Setup** (one command to customize everything):
```bash
bash template_scripts/get_started/setup_project.sh "my-awesome-project"
```

## 🛠️ Setup

After creating your project from the template:

1. **Install dependencies:**
   ```bash
   uv sync --group dev
   ```

2. **Install in editable mode:**
   ```bash
   uv pip install -e .
   ```

3. **Initialize git:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit from template"
   ```
