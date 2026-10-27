#!/usr/bin/env bash
# Install Rodin 3.x + SMT solvers plugin for headless PO discharge.
# Rodin downloads: https://sourceforge.net/projects/rodin-b-sharp/
# After install: Help → Install New Software → Rodin update site → "SMT Solvers" and "Camille" (text editor).
# Then set eventb.rodin_path in configs/default.yaml to the rodin executable.
set -euo pipefail
echo "See comments in this file; Rodin is not pip-installable."
