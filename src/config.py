from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    base_currency: str = "EUR" # Default base currency
    cache_ttl_min: int = 60 # Default streamlit data cache in minutes
    data_source : str = "ECB" # whether the data should be downloaded live or restored from online snapshot
    snapshot_path: str = "data/snapshot.parquet" #Default path to store data

    class Config:
        env_file = ".env"

settings = Settings()