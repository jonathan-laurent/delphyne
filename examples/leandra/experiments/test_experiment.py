import learning_experiments as le

if __name__ == "__main__":
    experiment = le.make_experiment(
        le.ExperimentSettings(
            output="test-output",
            training_problems=[],
            testing_problems=[],
            max_dollars_per_training_problem=0.5,
            max_requests_per_training_problem=30,
            max_dollars_per_testing_problem=0.2,
            max_requests_per_testing_problem=10,
        )
    )
    experiment.run_cli()
