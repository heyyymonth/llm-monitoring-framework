# 🚀 CI/CD Setup Guide

## Overview

This document provides step-by-step instructions for setting up the complete CI/CD pipeline for the LLM Performance Monitoring Framework.

## 📋 Prerequisites

- GitHub repository with admin access
- Docker Hub account (for container registry)
- PyPI account (for package publishing)

## 🔧 GitHub Repository Configuration

### 1. Enable GitHub Actions

1. Go to **Settings** → **Actions** → **General**
2. Set **Actions permissions** to "Allow all actions and reusable workflows"
3. Set **Workflow permissions** to "Read and write permissions"
4. Check "Allow GitHub Actions to create and approve pull requests"

### 2. Configure Branch Protection Rules

Navigate to **Settings** → **Branches** → **Add rule**

**Rule for `main` branch:**
```
Branch name pattern: main

Protection settings:
☑️ Require a pull request before merging
  ☑️ Require approvals (1)
  ☑️ Dismiss stale PR approvals when new commits are pushed
  ☑️ Require review from code owners

☑️ Require status checks to pass before merging
  ☑️ Require branches to be up to date before merging
  Required status checks:
    - test (ubuntu-latest, 3.8)
    - test (ubuntu-latest, 3.9) 
    - test (ubuntu-latest, 3.10)
    - test (ubuntu-latest, 3.11)
    - Build Validation
    - Security Scan

☑️ Require conversation resolution before merging
☑️ Include administrators
```

### 3. Required Secrets

Add the following secrets in **Settings** → **Secrets and variables** → **Actions**:

#### Repository Secrets

| Secret Name | Description | Example Value |
|-------------|-------------|---------------|
| `PYPI_API_TOKEN` | PyPI API token for package publishing | `pypi-AgEIcHl...` |
| `DOCKER_USERNAME` | Docker Hub username | `yourusername` |
| `DOCKER_PASSWORD` | Docker Hub password or access token | `dckr_pat_xxx` |
| `CODECOV_TOKEN` | Codecov upload token (optional) | `xxx-xxx-xxx` |

#### How to Get These Tokens

**PyPI API Token:**
1. Go to [PyPI Account Settings](https://pypi.org/manage/account/)
2. Create API token with scope "Entire account"
3. Copy the token (starts with `pypi-`)

**Docker Hub Token:**
1. Go to [Docker Hub Security](https://hub.docker.com/settings/security)
2. Create new access token
3. Copy the token

## 🏗️ Workflow Overview

### CI Workflow (`.github/workflows/ci.yml`)

**Triggers:**
- Pull requests to `main` or `develop`
- Pushes to `main`

**Jobs:**
1. **Test Suite** - Runs on Python 3.8-3.11 matrix
   - Code formatting checks (Black, isort)
   - Linting (flake8)
   - Unit tests with coverage
   - Integration tests
   - Package installation test

2. **Build Validation** - Verifies package builds correctly
3. **Security Scan** - Runs security audits
4. **Docker Build Test** - Tests containerization

### CD Workflow (`.github/workflows/cd.yml`)

**Triggers:**
- Pushes to `main`
- Version tags (`v*`)

**Jobs:**
1. **Final Tests** - Full test suite on merge
2. **Build and Publish** - PyPI package publishing (tags only)
3. **Docker Deploy** - Container registry publishing
4. **Create Release** - GitHub release creation (tags only)
5. **Deploy Docs** - API documentation deployment

### Dependencies Workflow (`.github/workflows/dependencies.yml`)

**Triggers:**
- Weekly schedule (Mondays 9 AM UTC)
- Manual dispatch

**Jobs:**
1. **Security Audit** - Weekly vulnerability scans
2. **Update Dependencies** - Automated dependency updates
3. **Dependency Review** - PR-based dependency analysis

## 🔒 Security Configuration

### Dependabot Setup

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    labels:
      - "dependencies"
    commit-message:
      prefix: "chore"
      include: "scope"
```

### CodeQL Analysis

Enable in **Settings** → **Security** → **Code scanning**:
1. Set up CodeQL analysis
2. Default configuration is sufficient

## 🚀 Release Process

### Automated Release (Recommended)

1. **Create and merge PR** with your changes
2. **Create version tag** on main branch:
   ```bash
   git checkout main
   git pull origin main
   git tag v1.0.0
   git push origin v1.0.0
   ```
3. **GitHub Actions automatically**:
   - Runs full test suite
   - Builds and publishes to PyPI
   - Creates Docker images
   - Generates GitHub release
   - Deploys documentation

### Manual Release

If you need to publish manually:

```bash
# Build package
python -m build

# Upload to PyPI
twine upload dist/*

# Build and push Docker image
docker build -t yourusername/llm-monitoring-framework:v1.0.0 .
docker push yourusername/llm-monitoring-framework:v1.0.0
```

## 📊 Monitoring CI/CD

### Status Badges

Add to README.md:

```markdown
[![CI Status](https://github.com/yourusername/llm-monitoring-framework/workflows/CI%20-%20Pull%20Request%20Validation/badge.svg)](https://github.com/yourusername/llm-monitoring-framework/actions)
[![PyPI version](https://badge.fury.io/py/llm-monitoring-framework.svg)](https://badge.fury.io/py/llm-monitoring-framework)
[![Docker Pulls](https://img.shields.io/docker/pulls/yourusername/llm-monitoring-framework)](https://hub.docker.com/r/yourusername/llm-monitoring-framework)
```

### Notifications

Configure in **Settings** → **Notifications**:
- Email notifications for failed workflows
- Slack/Discord webhooks for releases

## 🛠️ Troubleshooting

### Common Issues

**Test failures in CI but pass locally:**
- Check Python version differences
- Verify all dependencies in requirements.txt
- Check environment variables

**Docker build fails:**
- Verify Dockerfile syntax
- Check base image availability
- Ensure all files are included in build context

**PyPI publishing fails:**
- Verify PYPI_API_TOKEN secret
- Check package name conflicts
- Ensure version number is unique

**Coverage reports missing:**
- Install pytest-cov: `pip install pytest-cov`
- Verify coverage configuration in pyproject.toml
- Check test runner arguments

### Getting Help

1. Check **Actions** tab for detailed logs
2. Review **Security** tab for vulnerability reports
3. Check **Insights** → **Dependency graph** for dependency issues

## 🎯 Best Practices

### Code Quality
- Always run tests locally before pushing
- Use pre-commit hooks for formatting
- Write meaningful commit messages
- Keep PRs focused and small

### Security
- Regularly update dependencies
- Review Dependabot PRs promptly
- Monitor security advisories
- Use least-privilege access tokens

### Performance
- Cache dependencies in workflows
- Use matrix builds efficiently
- Optimize Docker images with multi-stage builds
- Monitor workflow execution times

## 📈 Metrics and Analytics

The CI/CD pipeline provides:
- **Test coverage** reports and trends
- **Build performance** metrics
- **Security vulnerability** tracking
- **Dependency health** monitoring
- **Release frequency** analytics

Access these through:
- GitHub Actions insights
- Codecov dashboard (if configured)
- Docker Hub analytics
- PyPI project statistics 