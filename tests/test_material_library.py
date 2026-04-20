# tests/test_material_library.py
import io
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


@pytest.fixture
def temp_env(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        import database
        monkeypatch.setattr(database, "DB_PATH", os.path.join(td, "test.db"))
        database.init_db()
        monkeypatch.setenv("OSCE_STORAGE_DIR", os.path.join(td, "storage"))
        yield td


def test_save_material_dedupes_identical_content(temp_env):
    import material_library as ml
    data = b"rubric body"
    m1 = ml.save_material(
        "rubric",
        file=io.BytesIO(data), filename="r.xlsx",
        display_name="R1", assessment_type="uk_osce",
        uploaded_by="u@x.edu",
    )
    m2 = ml.save_material(
        "rubric",
        file=io.BytesIO(data), filename="r.xlsx",
        display_name="R2",
        assessment_type="uk_osce",
        uploaded_by="u2@x.edu",
    )
    assert m1.content_sha256 == m2.content_sha256
    assert m1.id == m2.id


def test_list_materials_filters(temp_env):
    import material_library as ml
    ml.save_material(
        "rubric", file=io.BytesIO(b"a"), filename="a.xlsx",
        display_name="A", assessment_type="uk_osce", uploaded_by="u@x",
    )
    ml.save_material(
        "answer_key", file=io.BytesIO(b"b"), filename="b.xlsx",
        display_name="B", assessment_type="kpsom_osce", uploaded_by="u@x",
    )
    assert len(ml.list_materials(kind="rubric")) == 1
    assert len(ml.list_materials(assessment_type="kpsom_osce")) == 1


def test_open_material_returns_bytes(temp_env):
    import material_library as ml
    m = ml.save_material(
        "rubric", file=io.BytesIO(b"hello"), filename="r.xlsx",
        display_name="R", assessment_type="uk_osce", uploaded_by="u@x",
    )
    with ml.open_material(m) as f:
        assert f.read() == b"hello"


def test_archive_material_soft_deletes(temp_env):
    import material_library as ml
    from identity import User
    m = ml.save_material(
        "rubric", file=io.BytesIO(b"x"), filename="r.xlsx",
        display_name="R", assessment_type="uk_osce", uploaded_by="u@x",
    )
    ml.archive_material(m.id, by=User(email="a@x", role="admin", session_id="s"))
    assert len(ml.list_materials(kind="rubric")) == 0
    assert len(ml.list_materials(kind="rubric", include_archived=True)) == 1
