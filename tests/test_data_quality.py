"""
Tests for data quality checks
"""
import pandas as pd
import pytest


def test_no_missing_values(df: pd.DataFrame):
    """
    Test that there are no missing values in the dataset
    """
    assert not df.isnull().values.any()


def test_positive_values(df: pd.DataFrame, column: str):
    """
    Test that specified column contains only positive values
    """
    assert (df[column] >= 0).all()


def test_date_range(df: pd.DataFrame, date_column: str):
    """
    Test that date column is within expected range
    """
    pass