import math


class EarlyStopping:

    def __init__(
        self,
        patience=7,
        min_delta=0.0001
    ):

        self.patience = patience
        self.min_delta = min_delta

        self.best_score = float("-inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, score):

        if not math.isfinite(score):
            raise ValueError("Invalid validation score")

        if score > self.best_score + self.min_delta:

            self.best_score = score
            self.counter = 0

        else:

            self.counter += 1

            if self.counter >= self.patience:

                self.should_stop = True

        return self.should_stop