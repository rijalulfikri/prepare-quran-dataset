"""Per-segment Moshaf attribute overrides shared by the training pipelines."""

from quran_transcript import MoshafAttributes

_TASHEEL = {"tasheel_or_madd": "tasheel"}

# moshaf id -> segment index -> attributes overriding the moshaf-level ones.
SPECIAL_SEGMENT_OVERRIDES: dict[str, dict[str, dict]] = {
    "6.0": {
        "010.0158": dict(_TASHEEL),
        "027.0107": dict(_TASHEEL),
        "010.0140": dict(_TASHEEL),
        "010.0233": dict(_TASHEEL),
        "012.0023": {"noon_tamnna": "rawm"},
        "026.0076": {"raa_firq": "tarqeeq"},
    }
}


def build_special_moshaf_attrs(
    moshaf_id_to_dict: dict[str, dict],
) -> dict[str, dict[str, MoshafAttributes]]:
    """Apply SPECIAL_SEGMENT_OVERRIDES on top of the given moshaf dicts.

    Moshaf ids absent from `moshaf_id_to_dict` are skipped.
    """
    return {
        moshaf_id: {
            segment: MoshafAttributes(**(moshaf_id_to_dict[moshaf_id] | override))
            for segment, override in segments.items()
        }
        for moshaf_id, segments in SPECIAL_SEGMENT_OVERRIDES.items()
        if moshaf_id in moshaf_id_to_dict
    }


def resolve_moshaf_attr(
    moshaf_id: str,
    segment_index: str,
    base: dict[str, MoshafAttributes],
    special: dict[str, dict[str, MoshafAttributes]],
) -> MoshafAttributes:
    """Segment-specific attributes if overridden, else the moshaf's base ones."""
    return special.get(moshaf_id, {}).get(segment_index, base[moshaf_id])
