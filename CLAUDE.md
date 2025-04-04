# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Run Jupyter notebooks: `jupyter notebook <notebook_path>`
- View notebook diff: `nbdiff <notebook1> <notebook2>`
- Install dependencies: `pip install -r requirements.txt`

## Code Style
- Import order: standard libraries, 3rd party libraries, local imports
- Use PyTorch conventions for ML code
- Variable naming: snake_case for variables, CamelCase for classes
- Prefer type hints where appropriate
- Use descriptive variable names for tensors (e.g., `inputs` not `x`)
- Maintain consistent docstrings for functions and classes

## Project Structure
- Keep notebooks in weekly folders (`week1/`, `week2/`, etc.)
- Store datasets in `data/` directory
- Use `tmp/` for temporary files or experimental notebooks

## Formatting
- Format code cells with 2-space indentation
- Limit line length to 80 characters where possible
- Add markdown cells to explain code sections