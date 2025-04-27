# Contributing to SonixMind

Thank you for your interest in contributing to SonixMind! This document outlines the process and guidelines for contributing to this project.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in the Issues section.
2. If not, create a new issue with a descriptive title and clear description.
3. Include steps to reproduce the bug, expected behavior, and actual behavior.
4. Add any relevant screenshots or logs.
5. Use the bug report template if available.

### Suggesting Enhancements

1. Check if the enhancement has already been suggested in the Issues section.
2. If not, create a new issue with a descriptive title and clear description.
3. Describe the current behavior and explain the behavior you'd like to see.
4. Explain why this enhancement would be useful.
5. Use the feature request template if available.

### Pull Requests

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes.
4. Add or update tests as needed.
5. Ensure all tests pass.
6. Update documentation if necessary.
7. Submit a pull request to the main repository.

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sonixmind.git
cd sonixmind
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install FFmpeg if not already installed:
   - **macOS**: `brew install ffmpeg`
   - **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
   - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH

5. Run the application:
```bash
streamlit run app.py
```

## Style Guide

- Follow PEP 8 style guidelines for Python code.
- Use meaningful variable and function names.
- Add docstrings to functions and classes.
- Keep functions small and focused on a single task.
- Add comments where necessary to explain complex logic.

## Testing

- Add tests for new features or bug fixes.
- Run tests before submitting a pull request.
- Test your changes on different platforms if possible.

## Documentation

- Update README.md if your changes affect usage.
- Add or update function/method docstrings.
- Create or update wiki pages if necessary.

## License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (MIT License).

## Questions?

If you have any questions or need help, please open an issue or contact the maintainers.

Thank you for contributing to SonixMind! 