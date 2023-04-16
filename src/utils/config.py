from pathlib import Path


class Config:
    """
    Holds configuration parameters
    """

    def __init__(self):
        self.name = "basic_unet"
        self.root_dir = (
            Path(__file__)
            .resolve()
            .parents[2]
            .joinpath("data")
            .joinpath("output")
            .joinpath("section_1")
            .joinpath("cleaned_data")
        )
        self.n_epochs = 50
        self.learning_rate = 0.0001
        self.batch_size = 64
        self.patch_size = 64
        self.test_results_dir = (
            Path(__file__)
            .resolve()
            .parents[2]
            .joinpath("data")
            .joinpath("output")
            .joinpath("section_2")
            .joinpath("test_results")
        )
