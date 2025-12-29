from src.currency_calculator import convert_currency

def test_conversion():
    rates = {"USD": 1.0, "EUR": 0.9}
    assert convert_currency(100, "USD", "EUR", rates) == 90