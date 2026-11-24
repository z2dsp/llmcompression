"""Independent training/evaluation sources shared by both compression trainers."""
from collections import Counter
import hashlib
import math
from pathlib import Path
import random

from .corpus import read_messages
from .training_corpus import load_prepared


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): h.update(block)
    return h.hexdigest()


class TrainingMessages:
    def __init__(self, factory, forbidden, deduplicate, max_bytes):
        self.factory,self.forbidden,self.deduplicate,self.max_bytes=factory,forbidden,deduplicate,max_bytes
        self.stats=Counter()
        self.count=sum(1 for _ in self._iterate(self.stats))
        if not self.count: raise ValueError('No training messages remain after verification-overlap and length filtering')

    def _iterate(self, stats):
        seen=set() if self.deduplicate else None
        for data in self.factory():
            stats['input_messages']+=1
            if not data:
                stats['empty_skipped']+=1; continue
            if self.max_bytes and len(data)>self.max_bytes:
                stats['oversized_skipped']+=1; continue
            if data in self.forbidden:
                stats['verification_overlap_skipped']+=1; continue
            if seen is not None:
                key=hashlib.sha256(data).digest()
                if key in seen:
                    stats['duplicates_skipped']+=1; continue
                seen.add(key)
            stats['messages']+=1; stats['utf8_bytes']+=len(data)
            yield data

    def __iter__(self): return self._iterate(Counter())
    def __len__(self): return self.count


def load_external(args):
    verification=Path(args.verification_input).resolve()
    if verification==args.input.resolve(): raise ValueError('Training and verification inputs must be different files')
    if not all(math.isfinite(value) and value>0 for value in (args.validation_fraction,args.test_fraction)):
        raise ValueError('Validation and test fractions must be positive')
    groups=Counter()
    empty=oversized=0
    for text in read_messages(verification,args.text_field):
        data=text.encode('utf-8')
        if not data: empty+=1
        elif args.max_message_bytes and len(data)>args.max_message_bytes: oversized+=1
        else: groups[data]+=1
    unique=list(groups)
    if len(unique)<2: raise ValueError('Verification corpus needs at least two distinct eligible messages')
    if args.split=='random': random.Random(args.seed).shuffle(unique)
    n=max(1,min(len(unique)-1,int(len(unique)*args.validation_fraction/(args.validation_fraction+args.test_fraction))))
    validation,test=unique[:n],unique[n:]
    if args.keep_duplicates:
        validation=[m for m in validation for _ in range(groups[m])]
        test=[m for m in test for _ in range(groups[m])]
    prepared=args.input.is_dir() or args.input.name=='manifest.json'
    if prepared:
        if args.limit_messages is not None or args.keep_duplicates:
            raise ValueError('Prepared splits are fixed; use --train-bytes for a development limit')
        sources,original=load_prepared(args.input,max_message_bytes=args.max_message_bytes)
        factory=lambda:iter(sources['train'])
        source=Path(original['prepared_manifest'])
    else:
        source=args.input
        def factory():
            for i,text in enumerate(read_messages(args.input,args.text_field)):
                if args.limit_messages is not None and i>=args.limit_messages: break
                yield text.encode('utf-8')
        original={}
    train=TrainingMessages(factory,set(groups),not args.keep_duplicates and not prepared,args.max_message_bytes)
    verification_report={'path':str(verification),'sha256':digest(verification),'unique_messages':len(unique),
        'empty_skipped':empty,'oversized_skipped':oversized,'validation_messages':len(validation),'test_messages':len(test),
        'split_strategy':args.split,'seed':args.seed,'note':'All eligible verification groups split between validation and test using normalized validation/test fractions; none are trained on.'}
    corpus={**original,'mode':'independent_verification','training_path':str(source.resolve()),
        'training_filter_counts':dict(train.stats),'verification':verification_report,
        'splits':{'train':{'messages':len(train),'utf8_bytes':train.stats['utf8_bytes']},
                  'validation':{'messages':len(validation),'utf8_bytes':sum(map(len,validation))},
                  'test':{'messages':len(test),'utf8_bytes':sum(map(len,test))}}}
    return {'train':train,'validation':validation,'test':test},corpus,source
