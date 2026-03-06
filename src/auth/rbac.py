"""
Role-Based Access Control (RBAC) for the Banking Knowledge AI System.

Three access levels govern document visibility:
  Level 1 — Public:       Accessible to all staff (general policies)
  Level 2 — Confidential: Risk, compliance, and treasury staff only
  Level 3 — Restricted:   Board members and C-suite only

Each document is mapped to exactly one level.  Users authenticate with a
profile that carries an access level.  A user at level N can see all
documents at levels 1 .. N.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


# ═══════════════════════════════════════════════════════════════════
# Access levels
# ═══════════════════════════════════════════════════════════════════

class AccessLevel(IntEnum):
    """Document / user clearance levels (higher = more access)."""
    PUBLIC = 1         # All bank staff
    CONFIDENTIAL = 2   # Risk, compliance, treasury
    RESTRICTED = 3     # Board, C-suite


# ═══════════════════════════════════════════════════════════════════
# Document → access-level mapping
# ═══════════════════════════════════════════════════════════════════

DOCUMENT_ACCESS_MAP: dict[str, AccessLevel] = {
    # ── Level 1 — Public (10 docs) ──────────────────────────────
    "Consumer_Protection_Policy_2025":     AccessLevel.PUBLIC,
    "Digital_Banking_Policy_2025":         AccessLevel.PUBLIC,
    "Data_Governance_Policy_2025":         AccessLevel.PUBLIC,
    "HR_Training_Policy_2025":             AccessLevel.PUBLIC,
    "Compliance_Monitoring_Program_2025":  AccessLevel.PUBLIC,
    "BCM_Policy_2025":                     AccessLevel.PUBLIC,
    "Outsourcing_Policy_2025":             AccessLevel.PUBLIC,
    "Payment_Systems_Policy_2025":         AccessLevel.PUBLIC,
    "Vendor_Risk_Management_2025":         AccessLevel.PUBLIC,
    "Internal_Audit_Charter_2025":         AccessLevel.PUBLIC,

    # ── Level 2 — Confidential (13 docs) ────────────────────────
    "FX_Policy_2025":                      AccessLevel.CONFIDENTIAL,
    "Credit_Risk_Policy_2025":             AccessLevel.CONFIDENTIAL,
    "Market_Risk_Framework_2025":          AccessLevel.CONFIDENTIAL,
    "Liquidity_Policy_2025":               AccessLevel.CONFIDENTIAL,
    "Investment_Policy_2025":              AccessLevel.CONFIDENTIAL,
    "Operational_Risk_Framework_2025":     AccessLevel.CONFIDENTIAL,
    "Capital_Adequacy_Policy_2025":        AccessLevel.CONFIDENTIAL,
    "IFRS9_ECL_Policy_2025":              AccessLevel.CONFIDENTIAL,
    "Climate_Risk_Framework_2025":         AccessLevel.CONFIDENTIAL,
    "Stress_Testing_Framework_2025":       AccessLevel.CONFIDENTIAL,
    "Treasury_Operations_Manual_2025":     AccessLevel.CONFIDENTIAL,
    "IT_Security_Policy_2025":             AccessLevel.CONFIDENTIAL,
    "Model_Risk_Management_2025":          AccessLevel.CONFIDENTIAL,

    # ── Level 3 — Restricted (7 docs) ───────────────────────────
    "AML_Policy_2025":                     AccessLevel.RESTRICTED,
    "Sanctions_Compliance_Policy_2025":    AccessLevel.RESTRICTED,
    "Fraud_Prevention_Policy_2025":        AccessLevel.RESTRICTED,
    "Governance_Policy_2025":              AccessLevel.RESTRICTED,
    "Recovery_Resolution_Plan_2025":       AccessLevel.RESTRICTED,
    "Remuneration_Policy_2025":            AccessLevel.RESTRICTED,
    "Related_Party_Policy_2025":           AccessLevel.RESTRICTED,
}


def get_access_level(doc_name: str) -> int:
    """Return the integer access level for a document (defaults to RESTRICTED)."""
    return int(DOCUMENT_ACCESS_MAP.get(doc_name, AccessLevel.RESTRICTED))


# ═══════════════════════════════════════════════════════════════════
# User profiles
# ═══════════════════════════════════════════════════════════════════

@dataclass
class UserProfile:
    """Authenticated user with a clearance level."""
    username: str
    display_name: str
    role: str
    access_level: AccessLevel

    @property
    def level_label(self) -> str:
        return {
            AccessLevel.PUBLIC: "🟢 Public",
            AccessLevel.CONFIDENTIAL: "🟡 Confidential",
            AccessLevel.RESTRICTED: "🔴 Restricted",
        }[self.access_level]


# Demo user database  (username → (password, profile))
_DEMO_USERS: dict[str, tuple[str, UserProfile]] = {
    "teller": (
        "teller123",
        UserProfile("teller", "Sara Ahmed — Branch Teller", "Branch Operations", AccessLevel.PUBLIC),
    ),
    "risk_analyst": (
        "risk123",
        UserProfile("risk_analyst", "Omar Hassan — Risk Analyst", "Risk Management", AccessLevel.CONFIDENTIAL),
    ),
    "compliance": (
        "comp123",
        UserProfile("compliance", "Nour Ali — Compliance Officer", "Compliance", AccessLevel.CONFIDENTIAL),
    ),
    "cro": (
        "cro123",
        UserProfile("cro", "Dr. Youssef Kamel — Chief Risk Officer", "C-Suite", AccessLevel.RESTRICTED),
    ),
}


def authenticate(username: str, password: str) -> Optional[UserProfile]:
    """Authenticate a user.  Returns profile on success, None on failure."""
    entry = _DEMO_USERS.get(username.lower().strip())
    if entry and entry[0] == password:
        return entry[1]
    return None


def can_access(user: UserProfile, doc_name: str) -> bool:
    """Check whether *user* is cleared to see *doc_name*."""
    required = DOCUMENT_ACCESS_MAP.get(doc_name, AccessLevel.RESTRICTED)
    return user.access_level >= required


def get_accessible_docs(user: UserProfile) -> list[str]:
    """Return the list of document names the user may access."""
    return sorted(
        doc for doc, level in DOCUMENT_ACCESS_MAP.items()
        if user.access_level >= level
    )
