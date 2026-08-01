"""Tests du module marketrisk.data (générateur synthétique et loaders offline)."""

from __future__ import annotations

import warnings

import numpy as np

from marketrisk.data import (
    DEFAULT_ASSET_NAMES,
    generate_synthetic_returns,
    load_embedded_csv,
    load_market_data,
    load_yfinance_prices,
    returns_to_prices,
)


class TestSyntheticGenerator:
    def test_shape_and_columns(self) -> None:
        df = generate_synthetic_returns(n_days=500, seed=1)
        assert df.shape == (500, len(DEFAULT_ASSET_NAMES))
        assert list(df.columns) == list(DEFAULT_ASSET_NAMES)

    def test_reproducible_with_seed(self) -> None:
        a = generate_synthetic_returns(n_days=200, seed=123)
        b = generate_synthetic_returns(n_days=200, seed=123)
        np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())

    def test_correlation_structure_approximately_recovered(self) -> None:
        df = generate_synthetic_returns(n_days=4000, seed=7)
        corr = df.corr()
        # Actions_EU / Actions_US fortement corrélées positivement par construction.
        assert corr.loc["Actions_EU", "Actions_US"] > 0.5
        # Obligations_Souv / Actions_EU corrélation négative par construction.
        assert corr.loc["Obligations_Souv", "Actions_EU"] < 0

    def test_fat_tails_present(self) -> None:
        from scipy import stats

        df = generate_synthetic_returns(n_days=2000, seed=9)
        for col in df.columns:
            excess_kurt = stats.kurtosis(df[col], fisher=True)
            assert excess_kurt > 0  # queues plus épaisses que la loi normale

    def test_rejects_non_stationary_garch_params(self) -> None:
        try:
            generate_synthetic_returns(garch_alpha=0.6, garch_beta=0.6)
            assert False, "devrait lever ValueError"
        except ValueError:
            pass


class TestOfflineRobustness:
    def test_yfinance_loader_returns_none_without_network_or_package(self) -> None:
        # Doit toujours retourner None proprement (pas d'exception) que le
        # package yfinance soit absent ou que le réseau échoue.
        result = load_yfinance_prices(["FAKE_TICKER_XYZ"], start="2020-01-01")
        assert result is None or hasattr(result, "shape")

    def test_load_market_data_falls_back_to_synthetic_without_network(self) -> None:
        df = load_market_data(use_live_data=False, n_days=100, seed=1)
        assert df.shape[0] == 100

    def test_load_market_data_with_live_flag_still_falls_back(self) -> None:
        # use_live_data=True mais réseau non garanti dans l'environnement de
        # test : le pipeline doit tout de même retourner des données
        # exploitables (repli synthétique).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            df = load_market_data(
                tickers=["THIS_TICKER_DOES_NOT_EXIST_12345"],
                use_live_data=True,
                n_days=100,
                seed=1,
            )
        assert df.shape[0] >= 1

    def test_embedded_csv_loads(self) -> None:
        df = load_embedded_csv()
        assert df.shape[0] > 0
        assert df.shape[1] == len(DEFAULT_ASSET_NAMES)


class TestPriceConversion:
    def test_returns_to_prices_starts_at_base(self) -> None:
        df = generate_synthetic_returns(n_days=50, seed=2)
        prices = returns_to_prices(df, start_price=100.0)
        assert np.allclose(prices.iloc[0].to_numpy(), 100.0)
        assert prices.shape[0] == df.shape[0] + 1
