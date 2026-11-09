from prepare_quran_dataset.zipformer import moshaf_ways as mw


def test_override_applies_only_to_listed_segment(moshaf_6_0_dict):
    base_d = {"6.0": moshaf_6_0_dict}
    base = {k: mw.MoshafAttributes(**v) for k, v in base_d.items()}
    special = mw.build_special_moshaf_attrs(base_d)
    assert (
        mw.resolve_moshaf_attr("6.0", "010.0158", base, special).tasheel_or_madd
        == "tasheel"
    )
    assert mw.resolve_moshaf_attr("6.0", "001.0001", base, special) == base["6.0"]


def test_unknown_moshaf_falls_back_to_base(moshaf_6_0_dict):
    base_d = {"1.0": moshaf_6_0_dict}
    base = {k: mw.MoshafAttributes(**v) for k, v in base_d.items()}
    special = mw.build_special_moshaf_attrs(base_d)  # has no "6.0" entry
    assert special == {}
    assert mw.resolve_moshaf_attr("1.0", "010.0158", base, special) == base["1.0"]


def test_other_overrides_apply(base_attrs, special_attrs):
    noon = mw.resolve_moshaf_attr("6.0", "012.0023", base_attrs, special_attrs)
    raa = mw.resolve_moshaf_attr("6.0", "026.0076", base_attrs, special_attrs)
    assert noon.noon_tamnna == "rawm"
    assert raa.raa_firq == "tarqeeq"
