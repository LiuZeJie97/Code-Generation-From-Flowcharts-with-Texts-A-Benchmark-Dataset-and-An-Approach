class EarlyStop(Exception):
    def __init__(self):
        super().__init__(self)

    def __str__(self):
        return "early stop!"


class MaxEpoch(Exception):
    def __init__(self):
        super().__init__(self)

    def __str__(self):
        return "max epoch!"