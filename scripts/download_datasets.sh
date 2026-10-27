#!/usr/bin/env bash
# Fetch the public policy datasets used in the evaluation into data/raw/, then convert them:
#     bash scripts/download_datasets.sh && python benchmark/convert_real.py
set -euo pipefail
mkdir -p data/raw && cd data/raw

echo "[1/4] XACML benchmark policies (iTrust, HL7, gpms, pluto, conference, fedora, kmarket, obligation, Simpson)"
echo "      from the XPA repository (Dianxiang Xu et al.), Experiments/<name>/<name>.xml"
mkdir -p xacml
[ -d xacml/xpa ] || git clone --depth 1 https://github.com/dianxiangxu/XPA.git xacml/xpa

echo "[2/4] AWS managed IAM policies (SummitRoute mirror, 600+ policies) + MAMIP action registry"
mkdir -p aws_iam
[ -d summitroute ] || git clone --depth 1 https://github.com/SummitRoute/aws_managed_policies.git summitroute
[ -d aws_iam/mamip ] || git clone --depth 1 https://github.com/z0ph/MAMIP.git aws_iam/mamip

echo "[3/4] Kubernetes bootstrap ClusterRoles (kubernetes/kubernetes testdata)"
mkdir -p k8s_rbac
[ -f k8s_rbac/cluster-roles.yaml ] || curl -sSL -o k8s_rbac/cluster-roles.yaml \
  https://raw.githubusercontent.com/kubernetes/kubernetes/master/plugin/pkg/auth/authorizer/rbac/bootstrappolicy/testdata/cluster-roles.yaml

echo "[4/4] Amazon Employee Access Challenge (Kaggle) — needs Kaggle CLI + accepted rules (not used in the paper runs)"
echo "      kaggle competitions download -c amazon-employee-access-challenge -p data/raw/amazon_access"
mkdir -p amazon_access
echo "done — now run: python benchmark/convert_real.py"
