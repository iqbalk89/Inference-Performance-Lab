"""Reusable visual-model blocks shared by all workbench diagrams."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .contracts import Component, ComponentKind, Connection, Metric, Position


class DiagramBlock(ABC):
    """A replaceable object capable of producing the stable visual contract."""

    @abstractmethod
    def component(self) -> Component:
        raise NotImplementedError


@dataclass(frozen=True)
class TensorBlock(DiagramBlock):
    block_id: str
    label: str
    summary: str
    position: Position
    metrics: tuple[Metric, ...] = ()

    def component(self) -> Component:
        return Component(
            self.block_id, self.label, ComponentKind.TENSOR, self.summary,
            self.position, self.metrics, lane="process",
        )


@dataclass(frozen=True)
class OperationBlock(DiagramBlock):
    block_id: str
    label: str
    summary: str
    position: Position
    metrics: tuple[Metric, ...] = ()

    def component(self) -> Component:
        return Component(
            self.block_id, self.label, ComponentKind.OPERATION, self.summary,
            self.position, self.metrics, lane="process",
        )


@dataclass(frozen=True)
class ResourceBlock(DiagramBlock):
    block_id: str
    label: str
    kind: ComponentKind
    summary: str
    position: Position
    metrics: tuple[Metric, ...] = ()

    def component(self) -> Component:
        return Component(
            self.block_id, self.label, self.kind, self.summary,
            self.position, self.metrics, lane="hardware",
        )


class Path:
    """Reusable constructors for semantically distinct diagram connections."""

    @staticmethod
    def logical(path_id: str, source: str, target: str, name: str) -> Connection:
        return Connection(path_id, source, target, name, category="logical")

    @staticmethod
    def transfer(
        path_id: str,
        source: str,
        target: str,
        name: str,
        *,
        badge: str,
        metrics: tuple[Metric, ...],
    ) -> Connection:
        return Connection(
            path_id, source, target, name, category="transfer",
            metrics=metrics, badge=badge,
        )

    @staticmethod
    def physical(
        path_id: str,
        source: str,
        target: str,
        name: str,
        *,
        badge: str = "",
        metrics: tuple[Metric, ...] = (),
    ) -> Connection:
        return Connection(
            path_id, source, target, name, direction="both",
            category="physical", metrics=metrics, badge=badge,
        )

    @staticmethod
    def mapping(path_id: str, source: str, target: str, name: str) -> Connection:
        return Connection(path_id, source, target, name, category="mapping")
