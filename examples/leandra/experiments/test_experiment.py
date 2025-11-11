import learning_experiments as le

NUM_PROBLEMS = 4

if __name__ == "__main__":
    experiment = le.make_experiment(
        le.ExperimentSettings(
            output="test-output",
            mode="sketch",
            training_problems=list(le.TRAINING_THEOREMS.keys())[:NUM_PROBLEMS],
            testing_problems=list(le.TESTING_THEOREMS.keys())[:NUM_PROBLEMS],
            max_dollars_per_training_problem=0.05,
            max_requests_per_training_problem=30,
            max_dollars_per_testing_problem=0.05,
            max_requests_per_testing_problem=30,
        )
    )
    experiment.run_cli()
