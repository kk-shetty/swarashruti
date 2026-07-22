# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-07-22

### Added

- Project repository initialised with `.gitignore` for Python, virtual environments, and common editor artifacts
- `pyproject.toml` with project metadata, dependencies, and uv-managed environment
- Standard folder scaffold (core/, tests/, docs/adr/, data/, models/) with `.gitkeep` placeholders
- `pytest` configured with dev dependency group, verbose output, and a passing placeholder test
- `ruff` configured for linting and formatting enforcement
- `pre-commit` hooks wired to run `ruff` and `pytest` on every commit
- `CHANGELOG.md` initiated following Keep a Changelog format with SemVer