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

## Report Content

Reports in this repo are for high-level (weekly/leadership) review:

- Lead with **descriptions of findings and decisions**, and **results as
  tables or visualizations** — not codebase structure or implementation
  detail. Mathematical formulations are welcome where they carry the idea.
- Result tables follow the target paper's format: only metrics the paper
  reports, all of the paper's baselines as rows, our models appended,
  best value per column in bold, eval-set caveats as footnotes.
- Put figures/plots in `assets/` and reference them relatively.
