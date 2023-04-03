class ExperimentLogger:
    def __init__(self, func) -> None:
        self.func = func

    def log(self, q, maxq):
        self.func()
        print("Queries answererd: " + str(q) + "/" + str(maxq), end = "\r")