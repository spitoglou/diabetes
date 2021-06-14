from src.bgc_providers.ohio_bgc_provider import OhioBgcProvider


def test_provider():
    provider = OhioBgcProvider()
    data = provider.get_glycose_levels()
    assert type(data) == list
    assert data[3].attrib['value'] == '112'


def test_stream():
    provider = OhioBgcProvider()
    stream = provider.simulate_glucose_stream()
    assert next(stream)['value'] == 101.0
    assert next(stream)['value'] == 98.0
