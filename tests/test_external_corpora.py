import json
from pathlib import Path
import pytest
from llm_compression.external_corpora import load_external
from llm_compression.bit_training import build_parser as bit_parser, main as bit_main
from llm_compression.nn_training import main as nn_main


def inputs(tmp_path):
    train=tmp_path/'train.jsonl'; verification=tmp_path/'mesh.jsonl'
    train.write_text(''.join(json.dumps({'body':text})+'\n' for text in ['train '+str(i) for i in range(5)]+['mesh 0','mesh 7','train 0']))
    verification.write_text(''.join(json.dumps({'body':'mesh '+str(i)})+'\n' for i in range(8)))
    return train,verification


@pytest.mark.parametrize('duplicates',[False,True])
def test_external_split_uses_all_training_rows_and_excludes_verification(tmp_path,duplicates):
    train,verification=inputs(tmp_path)
    args=bit_parser().parse_args([str(train),'--verification-input',str(verification)]+(['--keep-duplicates'] if duplicates else []))
    partitions,report,_=load_external(args)
    assert len(partitions['train'])==(6 if duplicates else 5)
    assert len(partitions['validation'])==len(partitions['test'])==4
    heldout=set(partitions['validation'])|set(partitions['test'])
    assert not set(partitions['validation']) & set(partitions['test'])
    assert not set(partitions['train']) & heldout
    assert list(partitions['train'])==list(partitions['train'])
    assert report['training_filter_counts']['verification_overlap_skipped']==2
    assert report['verification']['path']==str(verification.resolve())


@pytest.mark.parametrize('neural',[False,True])
def test_trainers_only_evaluate_external_messages(tmp_path,neural):
    train,verification=inputs(tmp_path)
    output=tmp_path/'model'
    common=[str(train),'--verification-input',str(verification),'--output',str(output),
            '--model-kib','4','--evaluation-messages','0','--benchmark-messages','0']
    if neural:
        pytest.importorskip('torch')
        result=nn_main(common+['--device','cpu','--epochs','1','--threads','1','--embedding-dim','2','--context-bytes','1'])
    else:
        result=bit_main(common+['--order','1','--workers','1','--strengths','16'])
    assert result==0
    report=json.loads((output/'report.json').read_text())
    assert report['test']['messages']==4
    assert report['benchmark']['messages']==4 and report['benchmark']['all_round_trips_exact']
    assert report['corpus']['splits']['train']['messages']==5
    assert report['corpus']['verification']['sha256']
    assert report['corpus']['mode']=='independent_verification'


def test_statistical_sweep_checks_verification_contents_on_resume(tmp_path):
    pytest.importorskip('matplotlib')
    from llm_compression.bit_sweep import main
    train,verification=inputs(tmp_path)
    args=[str(train),'--verification-input',str(verification),'--output',str(tmp_path/'sweep'),
          '--sizes-kib','4','--orders','1','--workers','1','--strengths','16','--benchmark-messages','1']
    assert main(args)==0
    assert main(args+['--resume'])==0
    verification.write_text(verification.read_text().replace('mesh 0','changed mesh'))
    with pytest.raises(SystemExit): main(args+['--resume'])


def test_nn_sweep_forwards_external_verification(tmp_path):
    from llm_compression.nn_sweep import run_command
    from llm_compression.nn_training import build_parser
    train,verification=inputs(tmp_path)
    args=build_parser(sweep=True).parse_args([str(train),'--verification-input',str(verification)])
    command=run_command(args,4,1,tmp_path/'out')
    parsed=build_parser().parse_args(command[3:])
    assert parsed.verification_input==verification


def test_external_prepared_training_uses_training_partition_only(tmp_path):
    from llm_compression.training_corpus import RecordWriter
    _,verification=inputs(tmp_path)
    prepared=tmp_path/'prepared'; prepared.mkdir()
    splits={}
    for name,bodies in [('train',[b'train A',b'mesh 0',b'train B']),('validation',[b'unused wikipedia validation']),('test',[b'unused wikipedia test'])]:
        writer=RecordWriter(prepared/(name+'.mccorpus'))
        for body in bodies: writer.write(body)
        stats=writer.close()
        splits[name]={**stats,'files':[{'path':name+'.mccorpus',**stats}]}
    (prepared/'manifest.json').write_text(json.dumps({'format':'mesh-compression-corpus-v1','source':{},'preparation':{'sample_bytes':160},'splits':splits}))
    args=bit_parser().parse_args([str(prepared),'--verification-input',str(verification)])
    partitions,report,_=load_external(args)
    assert list(partitions['train'])==[b'train A',b'train B']
    assert all(message.startswith(b'mesh ') for key in ('validation','test') for message in partitions[key])
    assert report['training_filter_counts']['verification_overlap_skipped']==1


def test_cuda_sweep_uses_and_hashes_verification_input(tmp_path):
    torch=pytest.importorskip('torch')
    pytest.importorskip('matplotlib')
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    from llm_compression.nn_sweep import main
    train,verification=inputs(tmp_path)
    args=[str(train),'--verification-input',str(verification),'--output',str(tmp_path/'cuda-sweep'),
          '--sizes-kib','4','--contexts','1','--embedding-dim','2','--epochs','1','--threads','1',
          '--evaluation-messages','0','--benchmark-messages','1']
    assert main(args)==0
    assert main(args+['--resume'])==0
    verification.write_text(verification.read_text().replace('mesh 0','different mesh'))
    with pytest.raises(SystemExit): main(args+['--resume'])
