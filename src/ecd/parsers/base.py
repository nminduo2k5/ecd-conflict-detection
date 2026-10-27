from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from ecd.core import Policy


class PolicyParser(ABC):
    """Step [1a]: normalise an external policy format into ecd.core.Policy."""

    @abstractmethod
    def parse(self, path: str | Path) -> Policy: ...


def load_any(path: str | Path) -> Policy:
    """Pick a parser by extension / content."""
    p = Path(path)
    if p.suffix == ".json":
        import json
        data = json.loads(p.read_text())
        if "rules" in data and "attributes" in data:
            return Policy.from_dict(data)           # native format
        if "Statement" in data:
            from .aws_iam import AWSIAMParser
            return AWSIAMParser().parse(p)
    if p.suffix in (".xml", ".xacml"):
        from .xacml import XACMLParser
        return XACMLParser().parse(p)
    if p.suffix in (".yaml", ".yml"):
        from .k8s_rbac import K8sRBACParser
        return K8sRBACParser().parse(p)
    raise ValueError(f"cannot infer policy format for {p}")
