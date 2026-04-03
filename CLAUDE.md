# Project Conventions

## Report Structure

When generating reports, always use the following folder structure:

```
<report-name>/
  report.md        # Main report content
  assets/          # Images, figures, and other supporting files
```

- The folder name should be kebab-case (e.g., `sota-robot-hands/`, `hamer-wilor-training-report/`)
- All images and figures go inside `assets/`, not in a separate top-level directory
- Reference assets with relative paths from report.md (e.g., `assets/image.png`)
- Do NOT place report files or images at the project root level
