import polars as pl
import os
os.environ["RUST_BACKTRACE"] = "1"


df = pl.DataFrame({
    'a':[
        [0.1],
        [0.3]
    ]
}, schema={'a':pl.List(pl.Float32)})

df.lazy().select(pl.col('a')+1).sink_parquet('sink_test.pq', statistics=False)
# df.lazy().select(pl.col('a')+pl.lit(1).cast(pl.Float32)).sink_parquet('sink_test.pq', statistics=False)

print(pl.read_parquet('sink_test.pq'))
