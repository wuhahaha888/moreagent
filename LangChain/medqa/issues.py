"""错误属于哪个层次，与医学证据支持状态相互独立。"""
from dataclasses import asdict, dataclass
import hashlib


@dataclass(frozen=True)
class Issue:
    category: str
    object_id: str
    reason: str
    recoverable: bool = True
    severity: str = "warning"

    @property
    def fingerprint(self):
        return hashlib.sha256((self.category + self.object_id + self.reason).encode()).hexdigest()[:20]

    def public(self):
        return {**asdict(self), "fingerprint": self.fingerprint}
