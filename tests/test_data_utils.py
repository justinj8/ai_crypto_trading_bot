import pandas as pd
from data_utils import add_technical_indicators

def test_add_technical_indicators():
    data = {
        'open': list(range(40)),
        'high': [x + 1 for x in range(40)],
        'low': [x - 1 for x in range(40)],
        'close': list(range(40)),
        'volume': [100] * 40,
    }
    df = pd.DataFrame(data)
    result = add_technical_indicators(df)
    for col in ['sma_20', 'rsi', 'macd', 'atr']:
        assert col in result.columns
        assert not result[col].isna().any()
