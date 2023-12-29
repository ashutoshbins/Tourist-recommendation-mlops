from zenml.core.parameters.base_parameter import BaseParameter
class ModelNameConfig(BaseParameter):
    """Model Configs"""
    model_name:str="EnsembleModel"
    