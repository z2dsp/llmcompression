import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pytest
from llm_compression import enwik8
from llm_compression.nn_model import export_model, shapes


def test_download_checks_original_bytes_and_reuses_verified_file(tmp_path, monkeypatch):
    content=b'<xml>hello</xml>\n'
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w') as z: z.writestr('enwik8',content)
    monkeypatch.setattr(enwik8,'SIZE',len(content))
    monkeypatch.setattr(enwik8,'MD5',hashlib.md5(content).hexdigest())
    monkeypatch.setattr(enwik8,'SHA1',hashlib.sha1(content).hexdigest())
    monkeypatch.setattr(enwik8.urllib.request,'urlopen',lambda *a,**k:io.BytesIO(archive.getvalue()))
    path=enwik8.download(tmp_path)
    assert path.read_bytes()==content
    monkeypatch.setattr(enwik8.urllib.request,'urlopen',lambda *a,**k:pytest.fail('must reuse dataset'))
    assert enwik8.download(tmp_path)==path
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'): enwik8.download(tmp_path)


@pytest.mark.parametrize('kind',['mcnn','pnn','mcblm'])
def test_benchmark_models_and_standard_codecs_use_same_prefix(tmp_path,kind):
    source=tmp_path/'input'
    content=(b'hello world\n'+bytes(range(256)))*2
    source.write_bytes(content)
    model=tmp_path/('model.'+kind)
    if kind=='mcblm':
        from llm_compression.bit_training import train_counts
        from llm_compression.bit_model import save_model
        counts,_=train_counts([content],0,0,1024,workers=1)
        save_model(model,order=0,slots=0,strength=1,counts=counts)
    else:
        arrays=[np.full(shape,0.1,dtype='float32') for shape in shapes(2,2,3)]
        export_model(model,arrays,2,2,3,'int8')
        if kind=='pnn':
            from pico.prepare_model import export
            source_model=tmp_path/'source.mcnn'
            model.rename(source_model)
            export(source_model,model)
    output=tmp_path/'output'
    assert enwik8.main(['--input',str(source),'--output',str(output),'--limit-bytes','100',
                        '--model-block-bytes','37','--model',str(model)])==0
    report=json.loads((output/'report.json').read_text())
    assert report['benchmarked_bytes']==100 and report['prefix_only']
    assert report['sha256']==hashlib.sha256(content[:100]).hexdigest()
    assert len(report['results'])==4
    for row in report['results']:
        assert row['round_trip_exact']
        assert row['compressed_bytes']==Path(row['artifact']).stat().st_size
        assert row['bits_per_byte']==8*row['compressed_bytes']/100
    row=report['results'][-1]
    assert row['compressed_bytes']==row['arithmetic_payload_bytes']+enwik8.HEADER.size+3*enwik8.BLOCK.size
    assert row['bytes_including_model']==row['compressed_bytes']+model.stat().st_size
    with pytest.raises(SystemExit):
        enwik8.main(['--input',str(source),'--output',str(output)])
