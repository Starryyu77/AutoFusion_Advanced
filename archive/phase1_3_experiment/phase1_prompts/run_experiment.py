#!/usr/bin/env python3
"""
Wrapper script to run Phase 1 experiment from the correct directory.
"""
import os
import sys
import subprocess
from pathlib import Path

# Get paths
script_dir = Path(__file__).parent.resolve()
experiment_dir = script_dir.parent
project_root = experiment_dir.parent

# Change to experiment directory
os.chdir(experiment_dir)

# Set environment
env = os.environ.copy()
env['PYTHONPATH'] = str(project_root)

# Build command
cmd = [sys.executable, str(script_dir / 'run_phase1.py')] + sys.argv[1:]

# Run
result = subprocess.run(cmd, env=env)
sys.exit(result.returncode)
