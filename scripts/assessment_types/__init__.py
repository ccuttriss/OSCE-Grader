"""Assessment type registry.

Provides a central registry of all available assessment types and a factory
function to instantiate them by type ID.
"""

from .uk_osce import UKOSCEType
from .kpsom_osce import KPSOMHandoffType
from .kpsom_documentation import KPSOMDocumentationType
from .kpsom_ethics import KPSOMEthicsType

REGISTRY: dict[str, type] = {
    "uk_osce": UKOSCEType,
    "kpsom_ipass": KPSOMHandoffType,
    "kpsom_documentation": KPSOMDocumentationType,
    "kpsom_ethics": KPSOMEthicsType,
}


def get_type(type_id: str):
    """Instantiate and return an AssessmentType by its type ID."""
    if type_id not in REGISTRY:
        raise ValueError(f"Unknown assessment type: {type_id}")
    return REGISTRY[type_id]()
