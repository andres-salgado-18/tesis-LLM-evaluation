
def run_experiments(df, pipeline, configs):
    results = {}

    for config in configs:
        print(f"Running {config.id()}")

        df_out, path = pipeline.run(df, config)

        results[config.id()] = {
            "df": df_out,
            "path": path
        }

    return results