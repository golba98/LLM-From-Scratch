"""Streaming audit, finite mixtures and conservative split screening."""

from collections import Counter
from types import SimpleNamespace
import json

import pytest

from llm_data.conversational_audit import audit_files, near_duplicate, instruction_tags
from llm_data.curated_chat import curated_records
from llm_data.chat_mixture import ManifestChatDataset, TokenQuotaSampler


def test_contextual_instruction_candidates_exclude_discussion_only():
    assert 'one_word' in instruction_tags([{'role':'user','content':'Reply with exactly one word.'}])
    assert not instruction_tags([{'role':'user','content':'A program prints one word in its output stream.'}])


def test_near_duplicates_require_substantial_length_and_overlap():
    text=' '.join('word'+str(i) for i in range(100))
    assert near_duplicate(text,text+' extra')
    assert not near_duplicate('What is your name?', 'What is your name?')
    assert not near_duplicate(text,' '.join('other'+str(i) for i in range(100)))


def test_curated_records_are_unique_bounded_and_review_labeled():
    from llm_data.conversational_audit import content_identity
    rows=curated_records()
    assert 500<=len(rows)<=600
    assert len({content_identity(row['messages']) for row in rows})==len(rows)
    assert any(row['review_status']=='pending_human' for row in rows)
    assert all(row['family'].startswith('train-') for row in rows)
    for row in rows:
        if row['category']=='arithmetic':
            assert row['messages'][-1]['content'].isdigit()


def test_token_sampler_reconstruction_no_replacement_and_exposure():
    data=SimpleNamespace(index=[SimpleNamespace(source='a' if i%2 else 'b',
        assistant_tokens=3 if i%2 else 30,tokens=35) for i in range(100)])
    kwargs=dict(seed=42,probabilities={'a':.5,'b':.5},bucket_size=4)
    sampler=TokenQuotaSampler(data,**kwargs)
    order=list(sampler)
    assert order==list(TokenQuotaSampler(data,**kwargs)) and len(order)==len(set(order))
    assert len(order)<100  # Minority token pool caps the useful window.
    totals=Counter()
    for i in order:
        totals[data.index[i].source]+=data.index[i].assistant_tokens
    assert abs(totals['a']-totals['b'])<=30
    assert list(TokenQuotaSampler(data,**kwargs))[5:]==order[5:]
    sampler.set_epoch(1)
    assert list(sampler)!=order
    with pytest.raises(ValueError):
        TokenQuotaSampler(data,seed=1,probabilities={'a':1})


def test_unreviewed_manifest_cannot_be_used_for_training(tmp_path):
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps({'format':'codexa-chat-mixture-v1','admitted_for_training':False}))
    with pytest.raises(ValueError,match='admission'):
        ManifestChatDataset(path,None)


def test_streaming_audit_detects_root_and_content_leakage(tmp_path):
    from test_chat_dataset import Tokenizer, write_records
    train=tmp_path/'train.jsonl'; validation=tmp_path/'validation.jsonl'
    write_records(train); validation.write_bytes(train.read_bytes())
    report=audit_files([train,validation],Tokenizer(),tmp_path/'ledger.sqlite',context=16)
    assert {row['kind'] for row in report['cross_split_overlap']}=={'root','content'}
    assert report['datasets'][0]['supervised_tokens']==30
    assert report['datasets'][0]['counts']['language/unknown']==10
    with pytest.raises(FileExistsError):
        audit_files([train],Tokenizer(),tmp_path/'ledger.sqlite')
