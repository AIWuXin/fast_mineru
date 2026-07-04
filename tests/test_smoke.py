"""冒烟测试：不需要 GPU/模型也能跑的纯结构检查。"""
from pathlib import Path

from fast_mineru.config import PipelineConfig


def test_config_resolve_defaults():
    cfg = PipelineConfig().resolve()
    assert cfg.mfr_decoder_init_engine.name == "decoder_init_fp16.engine"
    assert cfg.mfr_decoder_past_engine.name == "decoder_with_past_fp16.engine"
    assert cfg.crnn_engine.name == "crnn.engine"  # tf32 默认


def test_config_fp16_rec_engine():
    cfg = PipelineConfig(crnn_engine_precision="fp16").resolve()
    assert cfg.crnn_engine.name == "crnn_fp16.engine"


def test_cli_collect_pdfs(tmp_path: Path):
    from fast_mineru.cli import _collect_pdfs
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "note.txt").write_text("x")
    pdfs = _collect_pdfs(tmp_path)
    assert len(pdfs) == 2
    assert _collect_pdfs(tmp_path / "a.pdf") == [tmp_path / "a.pdf"]


def test_imports():
    import fast_mineru
    from fast_mineru import FastMineruPipeline, PipelineConfig
    from fast_mineru.engines import TRTEngine, MFRDecoderTRT
    assert fast_mineru.__version__
