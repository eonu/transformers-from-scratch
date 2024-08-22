import enum
import typing as t

import pydantic as pyd

__all__ = [
    "DecodingType",
    "GreedySearchParams",
    "BeamSearchParams",
    "TemperatureSamplingParams",
    "TopKSamplingParams",
    "NucleusSamplingParams",
    "DecodingParams",
]


class DecodingType(enum.StrEnum):
    GREEDY = enum.auto()
    BEAM = enum.auto()
    TEMPERATURE = enum.auto()
    TOP_K = enum.auto()
    NUCLEUS = enum.auto()


class BaseDecodingParams(pyd.BaseModel, frozen=True):
    _decoding_type: DecodingType
    max_length: pyd.PositiveInt


class GreedySearchParams(BaseDecodingParams):
    _decoding_type: DecodingType = pyd.PrivateAttr(DecodingType.GREEDY)


class BeamSearchParams(BaseDecodingParams):
    _decoding_type: DecodingType = pyd.PrivateAttr(DecodingType.BEAM)
    beam_width: pyd.PositiveInt


class TemperatureSamplingParams(BaseDecodingParams):
    _decoding_type: DecodingType = pyd.PrivateAttr(DecodingType.TEMPERATURE)
    temperature: pyd.NonNegativeFloat
    k: pyd.PositiveInt | None = None


class TopKSamplingParams(BaseDecodingParams):
    _decoding_type: DecodingType = pyd.PrivateAttr(DecodingType.TOP_K)
    k: pyd.PositiveInt


class NucleusSamplingParams(BaseDecodingParams):
    _decoding_type: DecodingType = pyd.PrivateAttr(DecodingType.NUCLEUS)
    p: t.Annotated[float, pyd.Field(ge=0, le=1)]


DecodingParams = t.Annotated[
    GreedySearchParams
    | BeamSearchParams
    | TemperatureSamplingParams
    | TopKSamplingParams
    | NucleusSamplingParams,
    pyd.Field(discriminator="_decoding_type"),
]
