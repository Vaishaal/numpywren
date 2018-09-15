class ControlPlaneException(Exception):
    def __init__(self, msg):
        super().__init__(msg)

class LambdaPackParsingException(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class LambdaPackBackendGenerationException(Exception):
    def __init__(self, msg):
        super().__init__(msg)

class LambdaPackTypeException(Exception):
    def __init__(self, msg):
        super().__init__(msg)

class LambdaPackTimeoutException(Exception):
    def __init__(self, msg):
        super().__init__(msg)

class LambdaPackRetriesExhaustedException(Exception):
    def __init__(self, msg):
        super().__init__(msg)
