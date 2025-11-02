import learning_experiments as le

if __name__ == "__main__":
    experiment = le.make_experiment(
        le.ExperimentSettings(
            output="output",
            training_problems=list(le.TRAINING_THEOREMS.keys()),
            testing_problems=list(le.TESTING_THEOREMS.keys()),
            max_dollars_per_training_problem=2,
            max_requests_per_training_problem=100,
            max_dollars_per_testing_problem=1,
            max_requests_per_testing_problem=50,
        )
    )
    experiment.run_cli()
