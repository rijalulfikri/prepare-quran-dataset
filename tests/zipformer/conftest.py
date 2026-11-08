import pytest

from prepare_quran_dataset.zipformer import moshaf_ways as mw


@pytest.fixture
def moshaf_6_0_dict():
    """Fresh copy of a real `moshaf_metadata` row (moshaf 6.0) of
    obadx/mualem-recitations-annotated."""
    return {
        "comments": "من طريق الشاطبية",
        "total_size_mb": 1739.8306884765625,
        "path": "dataset/6.0",
        "madd_monfasel_len": 4,
        "irkab_maana": "idgham",
        "yaa_ataan": "wasl",
        "madd_yaa_alayn_alharfy": 6,
        "reciter_id": 6,
        "name": "المصحف المعلم",
        "rewaya": "hafs",
        "sakt_marqdena": "waqf",
        "almusaytirun": "saad",
        "sakt_iwaja": "sakt",
        "is_downloaded": True,
        "madd_aared_len": 4,
        "takbeer": "no_takbeer",
        "raa_yasr": "tafkheem",
        "madd_mottasel_waqf": 4,
        "reciter_arabic_name": "أيمن رشدي سويد",
        "yabsut": "seen",
        "between_anfal_and_tawba": "waqf",
        "tasheel_or_madd": "madd",
        "bastah": "seen",
        "bimusaytir": "saad",
        "start_with_ism": "wasl",
        "publisher": "https://ar.islamway.net/collection/11899",
        "sakt_bal_ran": "sakt",
        "sakt_man_raq": "sakt",
        "madd_alleen_len": 4,
        "segmented_by": "sura",
        "idgham_nakhluqkum": "idgham_kamil",
        "noon_tamnna": "ishmam",
        "recitation_speed": "above_murattal",
        "sources": ["https://download.quran.islamway.net/archives/11899-hi.zip"],
        "specific_sources": [
            {
                "sura_or_aya_index": "75",
                "url": "https://cdns1.zekr.online/quran/10178/75/76.mp3",
            }
        ],
        "yalhath_dhalik": "waqf",
        "harakat_daaf": "fath",
        "raa_misr": "tafkheem",
        "raa_nudhur": "tafkheem",
        "is_complete": True,
        "reciter_english_name": "Ayman Rushdi Swaid",
        "alif_salasila": "wasl",
        "raa_firq": "tafkheem",
        "is_annotated": True,
        "noon_and_yaseen": "izhar",
        "missing_recitations": [],
        "meem_aal_imran": "waqf",
        "raa_alqitr": "tarqeeq",
        "sakt_maleeyah": "waqf",
        "alias_start_sec": 0.0,
        "ghonna_lam_and_raa": "no_ghonna",
        "madd_mottasel_len": 4,
        "saken_before_hamz": "tahqeek",
        "num_recitations": 114,
        "id": "6.0",
        "total_duraion_minutes": 2527.248291015625,
        "meem_mokhfah": "meem",
    }


@pytest.fixture
def base_attrs(moshaf_6_0_dict):
    return {"6.0": mw.MoshafAttributes(**moshaf_6_0_dict)}


@pytest.fixture
def special_attrs(moshaf_6_0_dict):
    return mw.build_special_moshaf_attrs({"6.0": moshaf_6_0_dict})


@pytest.fixture
def tokens():
    from prepare_quran_dataset.zipformer.tokens import PhonemeTokens

    return PhonemeTokens.from_vocab_json("vocab_streaming/vocab.json")


@pytest.fixture
def cfg():
    from prepare_quran_dataset.zipformer.config import ZipformerTrainConfig

    return ZipformerTrainConfig(
        train_moshaf_ids=["6.0"],
        test_moshaf_ids=["19.0"],
        min_noise_input_seconds=1.0,
        max_noise_input_seconds=2.0,
    )


@pytest.fixture
def make_cut_in_memory():
    """Factory: `MonoCut` over an in-memory 16 kHz recording of zeros with one
    supervision spanning the cut (`custom={"input_type": ...}`)."""
    import io

    import numpy as np
    import soundfile as sf
    from lhotse import MonoCut, Recording, SupervisionSegment

    def _make(id, duration, text, input_type="speech"):
        buf = io.BytesIO()
        sf.write(buf, np.zeros(round(duration * 16000), dtype="float32"), 16000, format="WAV")
        rec = Recording.from_bytes(buf.getvalue(), recording_id=id)
        sup = SupervisionSegment(
            id=id, recording_id=id, start=0.0, duration=rec.duration,
            text=text, custom={"input_type": input_type},
        )
        return MonoCut(
            id=id, start=0.0, duration=rec.duration, channel=0,
            recording=rec, supervisions=[sup],
        )

    return _make


def _build_data_dir(root, num_speech, seconds):
    """Data dir in Task 4's layout: `seconds`-long FLACs under `root/audio/...` and
    `cuts_{train,dev,noise_aug}.jsonl.gz` manifests with relative paths."""
    import io

    import numpy as np
    import soundfile as sf

    from prepare_quran_dataset.zipformer.data_prep import _write_cuts, make_cut, write_flac

    rng = np.random.default_rng(0)

    def flac(rel):
        # low-amplitude sine + noise (not silence) so fbank is well-defined
        n = seconds * 16000
        t = np.arange(n) / 16000
        wav = 0.1 * np.sin(2 * np.pi * rng.uniform(100, 400) * t)
        wav += 0.01 * rng.standard_normal(n)
        buf = io.BytesIO()
        sf.write(buf, wav.astype("float32"), 16000, format="WAV")
        return write_flac(buf.getvalue(), root / rel)

    def cut(rel, text, cut_id, input_type):
        return make_cut(rel, flac(rel), text, cut_id, {"input_type": input_type}, root)

    train = [
        cut(f"audio/moshaf_6.0/{i}.flac", "ءَلِف"[: 1 + i % 4], f"moshaf_6.0__{i}", "speech")
        for i in range(num_speech)
    ]
    train.append(cut("audio/noise/000000.flac", "", "noise__000000", "silence"))
    noise_aug = [
        cut(f"audio/noise/{i:06d}.flac", "", f"noise__{i:06d}", "silence") for i in (1, 2)
    ]
    _write_cuts(train, root / "cuts_train.jsonl.gz")
    _write_cuts(train, root / "cuts_dev.jsonl.gz")
    _write_cuts(noise_aug, root / "cuts_noise_aug.jsonl.gz")
    return root


@pytest.fixture(scope="session")
def tiny_data_dir(tmp_path_factory):
    """3 speech + 1 silence three-second cuts (train == dev) and 2 noise_aug cuts.

    Three seconds, not one: with `drop_last=True` and Lhotse's minimum of 2
    buckets, x3 speed-perturbed one-second cuts (~12 s) cannot fill a full
    batch at `max_duration=10`, so the loader would yield nothing."""
    return _build_data_dir(tmp_path_factory.mktemp("tiny_data"), num_speech=3, seconds=3)


@pytest.fixture(scope="session")
def tiny_data_dir_big(tmp_path_factory):
    """Same layout with 60 one-second cuts, so an epoch at `max_duration=4` has >= 10 batches."""
    return _build_data_dir(tmp_path_factory.mktemp("tiny_data_big"), num_speech=59, seconds=1)
