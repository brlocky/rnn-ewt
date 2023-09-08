

from .zigzag import ZigZag


def process_pivots(df):
    z = ZigZag(df)
    pivots = z.get_zigzag()

    column_name = "Pivot"
    df = df.assign(**{column_name: pivots})

    return df
