def convert_currency(amount:float, from_currency:str, to_currency:str, exchange_rates:dict) -> float:
    """
    Convert an amount from one currency to another using given exchange rates.

    :param amount: The amount of money to convert.
    :param from_currency: The currency code of the original currency.
    :param to_currency: The currency code of the target currency.
    :param exchange_rates: A dictionary with currency codes as keys and their exchange rates to a base currency as values.
    :return: The converted amount in the target currency.
    """
    if from_currency not in exchange_rates or to_currency not in exchange_rates:
        raise ValueError("Invalid currency code provided.")

    # Convert the amount to the base currency
    base_amount = amount / exchange_rates[from_currency]
    
    # Convert the base amount to the target currency
    converted_amount = base_amount * exchange_rates[to_currency]
    
    return converted_amount