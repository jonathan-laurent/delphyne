from code2inv_experiments import Config, make_code2inv_experiment, run_app

configs = [
    Config(
        model_name="gpt-4.1-nano",
        bench_name=bench,
        temp=1,
        num_concurrent=4,
        max_requests=4,
        seed=0,
    )
    for bench in ["1", "7"]
]

if __name__ == "__main__":
    exp = make_code2inv_experiment("test", configs)
    run_app(exp)
