import json
import subprocess
import sys

import pytest

import pdf_to_glb as c
from test_converter import fixture_bytes, glb_worlds, grouped_fixture_bytes, read_glb


@pytest.mark.parametrize('level,ratio,error', [('low', .5, .001), ('medium', .25, .005), ('high', .1, .01)])
def test_presets_and_overrides(level, ratio, error):
    assert c.lightweight_options(level) == dict(level=level, ratio=ratio, error=error, lock_border=False, merge_parts=False)
    assert c.lightweight_options(level, .8, 0, True, True) == dict(level=level, ratio=.8, error=0, lock_border=True, merge_parts=True)


@pytest.mark.parametrize('options', [dict(level='bad'), dict(ratio=.5), dict(error=.01), dict(lock_border=True), dict(merge_parts=True),
    dict(level='low', ratio=0), dict(level='low', ratio=-1), dict(level='low', ratio=1.1),
    dict(level='low', ratio=float('nan')), dict(level='low', ratio=float('inf')),
    dict(level='low', error=-1), dict(level='low', error=1.1), dict(level='low', error=float('nan'))])
def test_invalid_options_rejected_before_reading_source(tmp_path, options):
    with pytest.raises(ValueError):
        c.convert(tmp_path / 'missing.pdf', tmp_path / 'output.glb', **options)
    assert not list(tmp_path.iterdir())


def test_default_unchanged_and_small_parts_retained(tmp_path):
    source = tmp_path / 'input.u3d'; source.write_bytes(fixture_bytes(opacity=.35))
    original = tmp_path / 'original.glb'
    baseline = c.convert(source, original)
    default = tmp_path / 'default.glb'
    assert c.convert(source, default, level='none') == baseline
    assert default.read_bytes() == original.read_bytes()
    output = tmp_path / 'light.glb'
    report = c.convert(source, output, level='high')
    doc, _ = read_glb(output)
    assert report['lightweight']['original_triangles'] == report['lightweight']['output_triangles'] == 2
    assert report['triangles_in_scene'] == 2
    assert doc['nodes'][1]['mesh'] == doc['nodes'][2]['mesh']
    assert glb_worlds(doc)[2][:3, 3] == pytest.approx([2, 0, 0])
    assert doc['materials'][0]['pbrMetallicRoughness']['baseColorFactor'] == pytest.approx([1, 0, 0, .35])
    assert json.loads(output.with_suffix('.report.json').read_text()) == report


@pytest.mark.parametrize('failure', ['missing-node', 'failed-process'])
def test_failed_optimizer_preserves_existing_output_and_report(tmp_path, monkeypatch, failure):
    source = tmp_path / 'input.u3d'; source.write_bytes(fixture_bytes())
    output = tmp_path / 'model.glb'; output.write_bytes(b'previous')
    report = output.with_suffix('.report.json'); report.write_text('previous report')
    if failure == 'missing-node':
        monkeypatch.setattr(c.shutil, 'which', lambda name: None)
    else:
        monkeypatch.setattr(c.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 1, '', 'missing module'))
    with pytest.raises(ValueError, match='Node.js|missing module'):
        c.convert(source, output, level='medium')
    assert output.read_bytes() == b'previous'
    assert report.read_text() == 'previous report'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['input.u3d', 'model.glb', 'model.report.json']


def test_cli_options_and_failure(tmp_path):
    source = tmp_path / 'input.u3d'; source.write_bytes(fixture_bytes())
    command = [sys.executable, str(c.Path(c.__file__)), str(source), str(tmp_path / 'output.glb')]
    result = subprocess.run(command + ['--level', 'low', '--ratio', '.8', '--error', '0', '--lock-border'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / 'output.report.json').read_text())
    assert report['lightweight']['ratio'] == .8
    assert report['lightweight']['lock_border'] is True
    result = subprocess.run(command + ['--ratio', '.5'], capture_output=True, text=True)
    assert result.returncode == 1
    assert 'require --level' in result.stderr
    result = subprocess.run(command + ['--level', 'high', '--merge-parts'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / 'output.report.json').read_text())
    assert report['lightweight']['output_instances'] == 1
    assert report['visible_instances'] == report['glb_meshes'] == 1
    assert 'selection removed' in report['node_mapping']


def test_merge_parts_explicitly_flattens_assembly_without_moving_geometry(tmp_path):
    source = tmp_path / 'input.u3d'; source.write_bytes(grouped_fixture_bytes())
    baseline = c.convert(source, tmp_path / 'original.glb')
    output = tmp_path / 'merged.glb'
    report = c.convert(source, output, level='high', merge_parts=True)
    assert report['triangles_in_scene'] == baseline['triangles_in_scene']
    assert report['dimensions_m'] == pytest.approx(baseline['dimensions_m'], abs=1e-6)
    assert report['visible_instances'] == 1
    assert report['group_nodes'] == 0
