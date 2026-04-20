"""
Global configuration constants for all preprocessing pipelines.
"""

# ---------------------------------------------------------------------------
# Base directories
# ---------------------------------------------------------------------------
RAW_DIR = "data/raw/"
PROCESSED_DIR = "data/processed/"
ARFF_DIR = "data/arff/"  # consumed by Java/MOA

# ---------------------------------------------------------------------------
# Yahoo Finance — exactly 80 tickers across 7 sectors
# ---------------------------------------------------------------------------
YAHOO_TICKERS = [
    # Technology (15)
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "AMD", "INTC", "CRM", "ADBE", "NFLX", "PYPL", "ORCL", "SHOP",
    # Healthcare (10)
    "JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT", "MDT", "AMGN", "GILD",
    # Finance (10)
    "JPM", "BAC", "GS", "MS", "V", "MA", "BLK", "AXP", "C", "WFC",
    # Energy (10)
    "XOM", "CVX", "COP", "SLB", "EOG", "DVN", "MPC", "VLO", "OXY", "HAL",
    # Consumer (10)
    "WMT", "PG", "KO", "PEP", "COST", "MCD", "NKE", "SBUX", "TGT", "HD",
    # Industrials (10)
    "CAT", "BA", "HON", "UPS", "GE", "MMM", "LMT", "RTX", "DE", "UNP",
    # Other / Mixed (15)
    "DIS", "CMCSA", "T", "VZ", "NEE", "DUK", "AMT", "SPG", "BX", "APO",
    "BKNG", "CME", "ICE", "MSCI", "ZTS",
]

assert len(YAHOO_TICKERS) == 80, f"Expected 80 tickers, got {len(YAHOO_TICKERS)}"

YAHOO_START_DATE = "2015-01-01"
YAHOO_END_DATE = "2025-01-01"
YAHOO_INTERVAL = "1d"  # daily bars

# ---------------------------------------------------------------------------
# NYC Taxi & Limousine Commission (TLC)
# ---------------------------------------------------------------------------
# Exactly 20 zone IDs (LocationID from TLC data dictionary)
NYC_ZONES = [
    161,  # Midtown Center (manhattan_core)
    236,  # Upper East Side North (manhattan_core)
    237,  # Upper East Side South (manhattan_core)
    170,  # Murray Hill (manhattan_core)
    100,  # Garment District (manhattan_core)
    132,  # JFK Airport (airport)
    138,  # LaGuardia Airport (airport)
    7,    # Astoria (residential)
    37,   # Central Harlem (residential)
    79,   # East Village (residential)
    114,  # Inwood Hill Park (residential)
    148,  # Lower East Side (residential)
    42,   # Central Park (suburban)
    62,   # Crown Heights North (suburban)
    119,  # Kips Bay (suburban)
    168,  # Meatpacking/West Village West (suburban)
    169,  # Midwood (suburban)
    244,  # Washington Heights South (suburban)
    163,  # Midtown North (manhattan_core)
    230,  # Times Sq/Theatre District (manhattan_core)
]

assert len(NYC_ZONES) == 20, f"Expected 20 zones, got {len(NYC_ZONES)}"

# Zone type encoding: 0=manhattan_core, 1=airport, 2=residential, 3=suburban
ZONE_TYPE = {
    161: 0, 236: 0, 237: 0, 170: 0, 100: 0, 163: 0, 230: 0,  # manhattan_core
    132: 1, 138: 1,                                             # airport
    7: 2, 37: 2, 79: 2, 114: 2, 148: 2,                       # residential
    42: 3, 62: 3, 119: 3, 168: 3, 169: 3, 244: 3,             # suburban
}

# Geographic adjacency for neighbor_avg_demand feature
ZONE_NEIGHBORS: dict[int, list[int]] = {
    161: [163, 170, 230, 119],
    163: [161, 170, 230, 37, 114],
    170: [161, 163, 236, 237, 230],
    230: [161, 163, 170, 100],
    100: [161, 230, 119],
    236: [237, 170, 163],
    237: [236, 170, 148],
    132: [138],
    138: [132],
    7:   [37, 79, 62],
    37:  [7, 79, 163, 114],
    79:  [7, 37, 148, 100],
    114: [37, 163, 244],
    148: [79, 237, 119],
    42:  [62, 119, 168],
    62:  [7, 42, 119, 169],
    119: [42, 62, 161, 100, 148],
    168: [42, 62, 169],
    169: [62, 168, 244],
    244: [114, 169],
}

# NYC TLC data range: 2022-01 to 2024-12 → exactly 36 files
NYC_YEARS = [2022, 2023, 2024]
NYC_MONTHS = list(range(1, 13))
NYC_TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"

# ---------------------------------------------------------------------------
# NHTS (National Household Travel Survey)
# ---------------------------------------------------------------------------
NHTS_EDITIONS = [2009, 2017, 2022]

# Paths relative to RAW_DIR where NHTS CSVs are expected
NHTS_LOCAL_DIRS = {
    2009: "nhts/2009/",
    2017: "nhts/2017/",
    2022: "nhts/2022/",
}

# TRPTRANS codes to keep per target class
NHTS_PRIVATE_VEHICLE_CODES = list(range(1, 8))    # 1–7
NHTS_TRANSIT_CODES = list(range(10, 17))           # 10–16
NHTS_ACTIVE_CODES = list(range(17, 20))            # 17–19
NHTS_VALID_TRPTRANS = (
    NHTS_PRIVATE_VEHICLE_CODES + NHTS_TRANSIT_CODES + NHTS_ACTIVE_CODES
)
