import pandas as pd


def convert_cols_with_matching_colnames_to_datetime(
    df: pd.DataFrame,
    colname_substr: str,
) -> pd.DataFrame:
    """Convert columns that contain colname_substr in their name to datetimes.

    Args:
        df (pd.DataFrame): The dataframe to convert.
        colname_substr (str): Substring to match on.

    Returns:
        pd.DataFrame: The converted dataframe.
    """
    df.loc[:, df.columns.str.contains(colname_substr)] = df.loc[
        :,
        df.columns.str.contains(colname_substr),
    ].apply(
        pd.to_datetime,
    )
    return df


def str_to_df(string: str, convert_timestamp_to_datetime: bool = True) -> pd.DataFrame:
    """Convert a string to a dataframe.

    Args:
        string (str): String to convert to a dataframe. String should be a csv.
        convert_timestamp_to_datetime (bool, optional): Whether to convert
            timestamp columns to datetime. Defaults to True.

    Returns:
        pd.DataFrame: The dataframe
    """
    from io import StringIO

    df = pd.read_table(StringIO(string), sep=",", index_col=False)

    if convert_timestamp_to_datetime:
        df = convert_cols_with_matching_colnames_to_datetime(df, "timestamp")

    # Drop "Unnamed" cols
    return df.loc[:, ~df.columns.str.contains("^Unnamed")]