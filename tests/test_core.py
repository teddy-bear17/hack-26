import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np


def test_categorizer_tax():
    from src.categorizer import TransactionCategorizer
    cat = TransactionCategorizer()
    df = pd.DataFrame({'payment_purpose': ['Единый налоговый платеж', 'ЕНП', 'Оплата налогов']})
    result = cat.predict(df)
    assert all('Налог' in c for c in result['category_ru'])


def test_categorizer_construction():
    from src.categorizer import TransactionCategorizer
    cat = TransactionCategorizer()
    assert 'Стройматериал' in cat.predict_single('Покупка цемента и арматуры')


def test_find_gap():
    from src.risk_engine import find_cash_gaps
    dates = [f'2026-01-{i:02d}' for i in range(1, 11)]
    balances = [100000, 50000, -20000, -50000, 0, 10000, 20000, -1000, 5000, 10000]
    result = find_cash_gaps(dates, balances)
    assert result['has_gap'] is True
    assert result['date'].day == 3


def test_no_gap():
    from src.risk_engine import find_cash_gaps
    dates = [f'2026-01-{i:02d}' for i in range(1, 11)]
    balances = [100000] * 10
    assert find_cash_gaps(dates, balances)['has_gap'] is False


def test_tax_calendar():
    from src.tax_calendar import generate_tax_events
    events = generate_tax_events('2026-01-01', days_ahead=60, monthly_tax_amount=100000)
    assert len(events) >= 2


def test_regular_payments():
    from src.regular_payments import detect_regular_payments
    df = pd.DataFrame({
        'operation_date': pd.date_range('2026-01-01', periods=6, freq='MS'),
        'direction': ['Списание'] * 6,
        'amount': [100000] * 6,
        'counterparty_name': ['ООО Арендодатель'] * 6,
        'payment_purpose': ['Аренда помещения по договору 123'] * 6,
    })
    regulars = detect_regular_payments(df, min_occurrences=3)
    assert len(regulars) >= 1