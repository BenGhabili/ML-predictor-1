# tasks.py in your project root

from invoke import task

@task
def data(ctx):
    ctx.run("python modules/data_preparation.py")

@task
def prepare_minute_data(ctx, timeframe=1, atr=1.0):
    ctx.run(f"python modules/data_preparation_x_min.py --timeframe {timeframe} --atr-mult {atr}")

@task
def simple_train(ctx, timeframe=1):
    """
    Train a KNN model on the most recent prepared CSV for the given timeframe.
    """
    ctx.run(f"python modules/train.py --timeframe {timeframe}")

@task
def train(ctx, timeframe=1):
    # 1) Read the data‑list.csv here
    import pandas as pd
    from pathlib import Path
    df = pd.read_csv(Path("data/data-list.csv"))
    tf_str = f"{timeframe}min"
    df_tf = df[df['Timeframe'] == tf_str].sort_values('Created_At', ascending=False)
    if df_tf.empty:
        print(f"No files for {tf_str}")
        return

    # 2) Show top 3
    for i, row in enumerate(df_tf.head(3).itertuples()):
        print(f"[{i}] {row.Filename}  ({row.Created_At})")

    # 3) Prompt for choice
    # choice = ctx.prompt("Which file index?", default="0")
    choice = input("Which file index? [0]: ") or "0"
    idx = int(choice)
    chosen = df_tf.iloc[idx].Filename

    # 4) Finally call your script
    ctx.run(f"python modules/train.py --timeframe {timeframe} --input data/{chosen}")    

@task
def api(ctx):
    ctx.run("python api/app.py")