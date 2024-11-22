import os
import pandas as pd
import requests
from datetime import date, timedelta
from configparser import ConfigParser
import logging


class CurrencyDataDownloader:
    def __init__(self, startDate, endDate, currency_querried):
        """
        Initializes the CurrencyDataDownloader with optional parameters.
        Defaults to using the current date if startDate or endDate is not provided.
        """
        self.startDate = startDate
        self.endDate = endDate
        self.currency_querried = currency_querried
        self.get_downloadDetails()

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler()]
        )

    def get_downloadDetails(self):
        """
        Reads configuration details from config.ini to set main currency, API base URL,
        output directory, and default table.
        """
        config = ConfigParser()

        if not os.path.exists("config.ini"):
            raise FileNotFoundError("config.ini not found. Please ensure the file exists and is named correctly.")

        config.read("config.ini")

        # Fetch values with defaults if not specified in config
        self.main_currency = config['data'].get('main_currency', 'PLN')
        self.api_base_url = config['data'].get('api_base_url', "https://api.nbp.pl/api/exchangerates/rates")
        self.output_directory = config['data'].get('output_directory', 'data')
        self.default_table = config['data'].get('default_table', 'c')

    def download_and_save_data(self, currency, output_path):
        """
        Downloads exchange rate data from the NBP API and saves it to a CSV file.
        """
        os.makedirs(output_path, exist_ok=True)
        output_file_path = os.path.join(output_path, f'exchange_rate_data_{currency}.csv')

        data_list = []
        current_date = self.startDate
        logging.info(f"Starting data download for {currency} from {self.startDate} to {self.endDate}")

        while current_date <= self.endDate:
            api_url = f"{self.api_base_url}/{self.default_table}/{currency}/{current_date}/"
            response = requests.get(api_url)

            if response.status_code == 200:
                try:
                    data = response.json()
                    exchange_rate_bid = data['rates'][0]['bid']
                    exchange_rate_ask = data['rates'][0]['ask']

                    data_list.append({
                        'Date': current_date,
                        'Currency': currency,
                        'Ask': exchange_rate_ask,
                        'Bid': exchange_rate_bid
                    })
                    logging.info(f"Data retrieved for {current_date}: Ask={exchange_rate_ask}, Bid={exchange_rate_bid}")
                except (KeyError, IndexError) as e:
                    logging.error(f"Error parsing data for {current_date}: {e}")
            else:
                logging.warning(f"Failed to retrieve data for {current_date}. Status code: {response.status_code}")

            current_date += timedelta(days=1)

        # Save the data to a CSV file
        if data_list:
            df = pd.DataFrame(data_list)
            df.to_csv(output_file_path, index=False)
            logging.info(f"Data saved to {output_file_path}")
        else:
            logging.warning(f"No data downloaded for {currency}. File not created.")

    def download_all_currencies(self, output_path):
        """
        Downloads exchange rate data for all currencies specified in the config file and the queried currency.
        """
        # Download the default main currency data
        self.download_and_save_data(self.main_currency, output_path)

        # Download the queried currency data
        self.download_and_save_data(self.currency_querried, output_path)
