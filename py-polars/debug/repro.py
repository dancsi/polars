import polars as pl
import os

os.environ["RUST_BACKTRACE"] = "1"

df = pl.DataFrame({
    'a': [
        [0.1],
        [0.2]
    ],
    'b': [
        [0.3],
        [0.4]
    ]
}, schema={'a': pl.List(pl.Float32), 'b': pl.Array(pl.Float32, 1)})

res = df.lazy().select((pl.col('a') + 1).alias('list_sum'), (pl.col('b') + 1).alias('arr_sum'),
                       (pl.col('a') * 2).alias('list_prod'), (pl.col('b') * 2).alias('arr_prod'))

print(res.collect_schema())
print(res.collect().schema)
