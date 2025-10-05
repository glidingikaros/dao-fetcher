import importlib


def test_imports():
    assert importlib.import_module('src.fetch_offchain_proposals')
    assert importlib.import_module('src.fetch_offchain_votes')
    assert importlib.import_module('src.fetch_onchain_proposals')
    assert importlib.import_module('src.fetch_onchain_votes')


def test_flatteners_snapshot():
    mod = importlib.import_module('src.fetch_offchain_proposals')
    sample = {
        'id': 'p1', 'ipfs': 'ipfs', 'space': {'id': 's'}, 'author': 'a', 'network': 'n',
        'created': 1, 'start': 2, 'end': 3, 'snapshot': '4', 'type': 't', 'quorum': 0,
        'state': 'closed', 'flagged': False, 'votes': 10, 'scores_total': 100,
        'scores_state': 'final', 'scores_updated': 5, 'title': 'x'
    }
    flat = mod.flatten_proposal_data(sample)
    assert flat['id'] == 'p1'
    assert 'metadata' in flat


essential_node = {
    'id': '1', 'onchainId': '2', 'status': 'active',
    'creator': {'address': '0x'}, 'proposer': {'address': '0x2'},
    'governor': {'id': 'g', 'name': 'gn'}, 'organization': {'id': 'o', 'name': 'on'},
    'quorum': 0,
    'start': {'number': 1, 'timestamp': 1},
    'end': {'number': 2, 'timestamp': 2},
    'block': {'number': 3, 'timestamp': 3},
    'metadata': {'title': 't'}
}


def test_flatteners_tally():
    mod = importlib.import_module('src.fetch_onchain_proposals')
    flat = mod._flatten_proposal_node(essential_node)
    assert flat['id'] == '1'
    assert flat['onchain_id'] == '2'
